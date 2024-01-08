import json

from fastchat.model.model_adapter import BaseModelAdapter
from fastchat.conversation import register_conv_template, Conversation, SeparatorStyle, get_conv_template

few_shot_question_template = 'Given the contexts: {context}, please answer: {question} '

incontext_few_shots = [{"context": "dummydata", "question": "dummyquestion", "answer": "dummyanswer"}, {"context": "dummydata", "question": "dummyquestion", "answer": "dummyanswer"}]
messages = [[("Human", few_shot_question_template.format(context=item['context'], question=item["question"])), ("Assistant", item["answer"])] for item in incontext_few_shots]
messages = tuple(sum(messages, []))

register_conv_template(
    Conversation(
        name="custom_few_shot",
        system_message="A chat between a curious human and an artificial intelligence clinician. "
        "The assistant gives helpful and detailed answers to the human's questions related to biomedicine.",
        roles=("Human", "Assistant"),
        messages=messages,
        offset=2,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)

register_conv_template(
    Conversation(
        name="pretrain_few_shot",
        system_message="",
        roles=("Question", "Answer"),
        messages=messages,
        offset=2,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)

register_conv_template(
    Conversation(
        name="one_shot",
        system_message="A chat between a curious human and an artificial intelligence clinician. "
        "The assistant gives helpful and detailed answers to the human's questions related to biomedicine.",
        roles=("Human", "Assistant"),
        messages=messages[:2],
        offset=2,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    ),
    override=True
)

register_conv_template(
    Conversation(
        name="eevee",
        system_message="Below is a health record paired with a question that describes a task.  Write a response that appropriately answers the question.",
        roles=("### Health Record", "### Answer"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n\n",
        sep2="</s>",
    )
)

class EeveeAdapter(BaseModelAdapter):

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "eevee" in model_path.lower()

    def get_default_conv_template(self, model_path: str):
        return get_conv_template("eevee")

class FewShotAdapter(BaseModelAdapter):

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "fewshot" in model_path.lower() and "pretrain" not in model_path.lower()

    def get_default_conv_template(self, model_path: str):
        return get_conv_template("custom_few_shot")

class PretrainFewShotAdapter(BaseModelAdapter):

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "pretrainfewshot" in model_path.lower()

    def get_default_conv_template(self, model_path: str):
        return get_conv_template("pretrain_few_shot")
