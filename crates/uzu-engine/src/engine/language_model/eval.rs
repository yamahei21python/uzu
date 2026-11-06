use std::range::Range;

use half::bf16;

use crate::{
    backends::common::{
        Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
        gpu_types::trie::TrieNode as GpuTrieNode,
    },
    data_type::DataType,
    encodable_block::batch_topology::BatchTopology,
    engine::language_model::{
        LanguageModel,
        state::LanguageModelState,
        stream::LanguageModelStreamError,
    },
    trie::TrieNode,
};

impl<B: Backend> LanguageModel<B> {
    /// Teacher-forced logits for a single sequence.
    ///
    /// The sequence is evaluated as a prefill batch; all positions are projected
    /// through the readout. The caller owns the state, so this can be used for
    /// KL / PPL evaluation without going through the sampling loop.
    ///
    /// Limits:
    /// * `token_ids.len()` must be <= 1024 (the existing prefill chunk limit).
    /// * The state is advanced by `token_ids`; use a fresh state for independent
    ///   sequences, or pass the prefix state for continuation.
    pub async fn teacher_forced_logits(
        &self,
        token_ids: &[u64],
        state: &mut LanguageModelState<B>,
    ) -> Result<Vec<f32>, LanguageModelStreamError<B>> {
        const MAX_BATCH_SIZE: usize = 1024;

        if token_ids.is_empty() {
            return Err(LanguageModelStreamError::EmptyInput);
        }
        if token_ids.len() > MAX_BATCH_SIZE {
            return Err(LanguageModelStreamError::ContextOverflow);
        }
        if self
            .decoder
            .max_context_length()
            .is_some_and(|max| state.transformer_state.context_length() + token_ids.len() as u32 > max)
        {
            return Err(LanguageModelStreamError::ContextOverflow);
        }

        state
            .transformer_state
            .prepare(state.transformer_state.context_length(), token_ids.len() as u32)
            .map_err(LanguageModelStreamError::Backend)?;

        let mut command_buffer = self
            .engine
            .context
            .create_command_buffer(Some("eval-logits"), None)
            .map_err(LanguageModelStreamError::Backend)?;

        let token_buffer = command_buffer
            .allocate_constant_from_slice(
                &token_ids.iter().map(|token| *token as u32).collect::<Box<[u32]>>(),
            )
            .map_err(LanguageModelStreamError::Backend)?;

        let input_trie = TrieNode::flat(state.tokens.len(), token_ids, &state.prng);
        let flat_trie = input_trie.linearize();
        let trie_nodes = flat_trie.token_subtrie_ranges().collect::<Box<[GpuTrieNode]>>();
        let batch_topology = BatchTopology::new(&trie_nodes, true);

        let output_range: Range<u32> = (0..token_ids.len() as u32).into();
        let decoder_output = self.decoder.encode(
            &token_buffer,
            &batch_topology,
            Some(output_range.clone()),
            None,
            &mut state.transformer_state,
            &mut command_buffer,
        )?;

        let logits = decoder_output
            .logits
            .as_ref()
            .expect("output_range requires decoder logits");

        let row_count = (output_range.end - output_range.start) as usize;
        let bytes = row_count * self.vocab_size * DataType::BF16.size_in_bytes();
        let mut output = self
            .engine
            .context
            .create_buffer_gpu(bytes)
            .map_err(LanguageModelStreamError::Backend)?;
        command_buffer.encode_copy(logits, &mut output);

        state
            .transformer_state
            .encode_accept(
                &(0..token_ids.len() as u32).collect::<Box<[u32]>>(),
                &mut command_buffer,
            )
            .map_err(LanguageModelStreamError::Backend)?;
        state.tokens.extend_from_slice(token_ids);

        drop(decoder_output);
        drop(token_buffer);

        command_buffer
            .end_encoding()
            .submit()
            .wait_until_completed()
            .await
            .map_err(LanguageModelStreamError::Backend)?;

        let values = self
            .engine
            .context
            .copyout::<bf16>(&output)
            .await
            .map_err(LanguageModelStreamError::Backend)?;
        Ok(values.iter().map(|value| value.to_f32()).collect())
    }
}
