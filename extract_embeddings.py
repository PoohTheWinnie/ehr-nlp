from vllm import LLM, SamplingParams
from vllm.sequence import SequenceData, SequenceGroupMetadata
import torch

llm = LLM(model="/root/hdd/llm/opt-1.3b")

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Profile memory usage with max_num_sequences sequences and the total
# number of tokens equal to max_num_batched_tokens.

# Enable top-k sampling to reflect the accurate memory usage.
vocab_size = llm.llm_engine.workers[0].model.config.vocab_size
sampling_params = SamplingParams(top_p=0.99, top_k=vocab_size - 1)

prompt = 'I am a'

prompt_token_ids = llm.llm_engine.tokenizer.encode(prompt) #[2, 100, 524, 10]

seqs = []
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

input_tokens, input_positions, input_metadata = llm.llm_engine.workers[0]._prepare_inputs(seqs)

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