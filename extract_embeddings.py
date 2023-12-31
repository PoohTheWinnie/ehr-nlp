from vllm import LLM, SamplingParams
llm = LLM(model="/root/hdd/llm/opt-1.3b")
from vllm.sequence import (SamplerOutput, Sequence, SequenceGroup,
                           SequenceGroupMetadata, SequenceGroupOutputs,
                           SequenceOutputs, SequenceStatus)

prompt = 'I am a'

# prompt_token_ids = llm.llm_engine.tokenizer.encode(prompt)
# block_size = llm.llm_engine.cache_config.block_size
# seq_id = next(llm.llm_engine.seq_counter)
# seq = Sequence(seq_id, prompt, prompt_token_ids, block_size)
# request_id = str(next(llm.request_counter))
# # Create the sequence group.
# seq_group = SequenceGroup(request_id, [seq], sampling_params,
#                             None)
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata
import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Profile memory usage with max_num_sequences sequences and the total
# number of tokens equal to max_num_batched_tokens.

# Enable top-k sampling to reflect the accurate memory usage.
vocab_size = llm.llm_engine.workers[0].model.config.vocab_size
sampling_params = SamplingParams(top_p=0.99, top_k=vocab_size - 1)
max_num_batched_tokens = llm.llm_engine.workers[0].scheduler_config.max_num_batched_tokens
max_num_seqs = llm.llm_engine.workers[0].scheduler_config.max_num_seqs

prompt = 'I am a'

prompt_token_ids = llm.llm_engine.tokenizer.encode(prompt) #[2, 100, 524, 10]

seqs = []
# for group_id in range(max_num_seqs):
#     seq_len = (max_num_batched_tokens // max_num_seqs +
#                 (group_id < max_num_batched_tokens % max_num_seqs))
#     print(seq_len)
group_id = 1
seq_data = SequenceData(prompt_token_ids)
seq = SequenceGroupMetadata(
    request_id=str(group_id),
    is_prompt=True,
    seq_data={group_id: seq_data},
    sampling_params=sampling_params,
    block_tables=None,
)
seqs.append(seq)

input_tokens, input_positions, input_metadata = llm.llm_engine.workers[0]._prepare_inputs(
    seqs)
# Execute the model.
num_layers = llm.llm_engine.workers[0].model_config.get_num_layers(llm.llm_engine.workers[0].parallel_config)
tempOut = llm.llm_engine.workers[0].model.model(
    input_ids=input_tokens,
    positions=input_positions,
    kv_caches=[(None, None)] * num_layers,
    input_metadata=input_metadata,
    cache_events=None,
)
tempOut.size() #torch.Size([1, 4, 2048])