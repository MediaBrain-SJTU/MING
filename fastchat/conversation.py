import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any
import pdb


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    DOLLY = auto()
    CHATGLM = auto()
    DOCTOR = auto()
    BLOOM = auto()
    QWEN = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    # Used for gradio server
    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += self.sep + " " + role + ": " + message
                else:
                    ret += self.sep + " " + role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.BLOOM:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if i == 0 and "我" in message and "你好" not in message and "您好" not in message:
                        ret += role + ": 你好，" + message + seps[i % 2]
                    else:
                        ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":" 
            return ret
        elif self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": \n" + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += "\n\n"
                else:
                    ret += role + ": \n"
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += f"[Round {i//2}]\n"
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ": "
            return ret
        elif self.sep_style == SeparatorStyle.DOCTOR:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 1:
                    ret += f"\nRound {i//2+1}\n"
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.QWEN:
            seps = [self.sep, self.sep2]
            ret = f"<|im_start|>system\n{self.system}<|im_end|>\n"
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + "\n" + message + seps[i % 2]
                else:
                    ret += role + "\n"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


conv_one_shot = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "What are the key differences between renewable and non-renewable energy sources?"),
        ("Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)


conv_vicuna_v1_1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)


conv_koala_v1 = Conversation(
    system="BEGINNING OF CONVERSATION:",
    roles=("USER", "GPT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_dolly = Conversation(
    system=
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n",
    roles=('### Instruction', '### Response'),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.DOLLY,
    sep="\n\n",
    sep2="### End",
)

conv_bloom = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.BLOOM,
    sep=" ",
    sep2="</s>",
)

conv_chatglm = Conversation(
    system="",
    roles=('问', '答'),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.CHATGLM,
    sep="\n",
    sep2="\n",
)

conv_vicuna_doctor = Conversation(
    # system="You are a doctor online.\n You ask the user for some medical information and some medical test results and indicators to get full knowledge of user's condition. At each Round you asks a medical question.",
    system="Marv is a doctor online, Marv asks the Patient for some medical information and some medical test results and indicators to get full knowledge of patient’s condition.\nAt each Round Marv asks a medical question.\n",
    roles=("Patient", "Marv"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.DOCTOR,
    sep=" ",
    sep2="</s>",
)

conv_baichuan = Conversation(
    system="",
    roles=("问", "答"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_qwen = Conversation(
    system="You are a helpful assistant.",
    roles=("<|im_start|>user", "<|im_start|>assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.QWEN,
    sep="<|im_end|>\n",
    sep2="<|im_end|>\n",
)

conv_templates = {
    "conv_one_shot": conv_one_shot,
    "vicuna_v1.1": conv_vicuna_v1_1,
    "koala_v1": conv_koala_v1,
    "dolly": conv_dolly,
    "baichuan": conv_baichuan,
    "bloom": conv_bloom,
    "qwen": conv_qwen
}


def get_default_conv_template(model_name):
    model_name = model_name.lower()
    if "vicuna" in model_name or "output" in model_name:
        return conv_vicuna_v1_1
    elif "qwen" in model_name:
        return conv_qwen
    elif "baichuan" in model_name:
        # print("load conv_baichuan")
        return conv_baichuan
    elif "doctor" in model_name:
        return conv_vicuna_doctor
    elif "bloom" in model_name:
        return conv_bloom
    elif "koala" in model_name:
        return conv_koala_v1
    elif "dolly" in model_name:
        return conv_dolly
    elif "chatglm" in model_name:
        return conv_chatglm
    return conv_one_shot


if __name__ == "__main__":
    print(default_conversation.get_prompt())
