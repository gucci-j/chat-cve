# See https://github.com/huggingface/lighteval/wiki/Adding-a-Custom-Task

import numpy as np
from aenum import extend_enum

from lighteval.metrics.metrics import Metrics, SampleLevelMetric
from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# DEFINE YOUR PROMPT FUNCTIONS
# Define as many as you need for your different tasks
def prompt_fn(
    line, 
    task_name: str = None
):
    """Defines how to go from a dataset line to a doc object.
    Follow examples in src/lighteval/tasks/tasks_prompt_formatting.py, or get more info
    about what this function should do in the README.
    """
    task_name = task_name.split("|")[1]
    if task_name == "mt:en2my":
        return Doc(
            task_name=task_name,
            query=f"အင်္ဂလိပ်မှ မြန်မာသို့ ဘာသာပြန်ပါ။:\n{line['en']} =",
            gold_index=0,
            choices=[line["my"]],
            instruction="အင်္ဂလိပ်မှ မြန်မာသို့ ဘာသာပြန်ပါ။:\n",
        )
    elif task_name == "mt:my2en":
        return Doc(
            task_name=task_name,
            query=f"မြန်မာမှ အင်္ဂလိပ်သို့ ဘာသာပြန်ပါ။:\n{line['my']} =",
            gold_index=0,
            choices=[line["en"]],
            instruction="မြန်မာမှ အင်္ဂလိပ်သို့ ဘာသာပြန်ပါ။:\n",
        )
    elif task_name == "mt:en2si":
        return Doc(
            task_name=task_name,
            query=f"ඉංග්‍රීසි සිංහලයට පරිවර්තනය කරන්න:\n{line['en']} =",
            gold_index=0,
            choices=[line["si"]],
            instruction="ඉංග්‍රීසි සිංහලයට පරිවර්තනය කරන්න:\n",
        )
    elif task_name == "mt:si2en":
        return Doc(
            task_name=task_name,
            query=f"සිංහලයෙන් ඉංග්‍රීසියට පරිවර්තනය කරන්න:\n{line['si']} =",
            gold_index=0,
            choices=[line["en"]],
            instruction="සිංහලයෙන් ඉංග්‍රීසියට පරිවර්තනය කරන්න:\n",
        )
    elif task_name == "mt:en2te":
        return Doc(
            task_name=task_name,
            query=f"ఆంగ్లం నుండి తెలుగుకు అనువదించండి:\n{line['en']} =",
            gold_index=0,
            choices=[line["te"]],
            instruction="ఆంగ్లం నుండి తెలుగుకు అనువదించండి:\n",
        )
    elif task_name == "mt:te2en":
        return Doc(
            task_name=task_name,
            query=f"తెలుగు నుండి ఆంగ్లంకు అనువదించండి:\n{line['te']} =",
            gold_index=0,
            choices=[line["en"]],
            instruction="తెలుగు నుండి ఆంగ్లంకు అనువదించండి:\n",
        )
    elif task_name == "mt:en2ta":
        return Doc(
            task_name=task_name,
            query=f"ஆங்கிலத்திலிருந்து தமிழுக்கு மொழிபெயர்க்கவும்:\n{line['en']} =",
            gold_index=0,
            choices=[line["ta"]],
            instruction="ஆங்கிலத்திலிருந்து தமிழுக்கு மொழிபெயர்க்கவும்:\n",
        )
    elif task_name == "mt:ta2en":
        return Doc(
            task_name=task_name,
            query=f"தமிழிலிருந்து ஆங்கிலத்திற்கு மொழிபெயர்க்கவும்:\n{line['ta']} =",
            gold_index=0,
            choices=[line["en"]],
            instruction="தமிழிலிருந்து ஆங்கிலத்திற்கு மொழிபெயர்க்கவும்:\n",
        )
    elif task_name == "mt:en2am":
        return Doc(
            task_name=task_name,
            query=f"እንግሊዝኛን ወደ አማርኛ ተርጉም:\n{line['en']} =",
            gold_index=0,
            choices=[line["am"]],
            instruction="እንግሊዝኛን ወደ አማርኛ ተርጉም:\n",
        )
    elif task_name == "mt:am2en":
        return Doc(
            task_name=task_name,
            query=f"አማርኛን ወደ እንግሊዝኛ ተርጉም:\n{line['am']} =",
            gold_index=0,
            choices=[line["en"]],
            instruction="አማርኛን ወደ እንግሊዝኛ ተርጉም:\n",
        )
    elif task_name == "mt:en2bn":
        return Doc(
            task_name=task_name,
            query=f"ইংরেজি থেকে বাংলায় অনুবাদ করুন:\n{line['en']} =",
            gold_index=0,
            choices=[line["bn"]],
            instruction="ইংরেজি থেকে বাংলায় অনুবাদ করুন:\n",
        )
    elif task_name == "mt:bn2en":
        return Doc(
            task_name=task_name,
            query=f"বাংলা থেকে ইংরেজিতে অনুবাদ করুন:\n{line['bn']} =",
            gold_index=0,
            choices=[line["en"]],
            instruction="বাংলা থেকে ইংরেজিতে অনুবাদ করুন:\n",
        )
    elif task_name == "mt:en2gu":
        return Doc(
            task_name=task_name,
            query=f"અંગ્રેજીમાંથી ગુજરાતીમાં અનુવાદ કરો:\n{line['en']} =",
            gold_index=0,
            choices=[line["gu"]],
            instruction="અંગ્રેજીમાંથી ગુજરાતીમાં અનુવાદ કરો:\n",
        )
    elif task_name == "mt:gu2en":
        return Doc(
            task_name=task_name,
            query=f"ગુજરાતીમાંથી અંગ્રેજીમાં અનુવાદ કરો:\n{line['gu']} =",
            gold_index=0,
            choices=[line["en"]],
            instruction="ગુજરાતીમાંથી અંગ્રેજીમાં અનુવાદ કરો:\n",
        )
    else:
        raise NotImplementedError


# CUSTOM METRIC IF NEEDED
class SampleLevelTranslationMetric:
    def __init__(self, metric_type: str):
        """Stores the relevant parameters for a corpus level translation metric.

        Args:
            metric_type (str): Can be any of bleu, chrf, or ter depending on the metric to use.
        """
        import sacrebleu
        if metric_type == "bleu":
            self.metric = sacrebleu.sentence_bleu
        elif metric_type == "chrf":
            self.metric = sacrebleu.sentence_chrf
        elif metric_type == "ter":
            self.metric = sacrebleu.sentence_ter
        else:
            raise ValueError(f"Unknown corpus level translation metric type : {metric_type}")

    def compute(self, golds: list[str], predictions: list[str], **kwargs) -> float:
        assert len(golds) == 1 and len(predictions) == 1
        return float(self.metric(predictions.pop(), golds).score)

chrf_sample = SampleLevelMetric(
    metric_name="chrf_sample",
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.TRANSLATION,
    sample_level_fn=SampleLevelTranslationMetric("chrf").compute, # how to compute score for one sample
    corpus_level_fn=np.mean, # aggregation
    higher_is_better=True,
)
extend_enum(Metrics, "chrf_sample", chrf_sample)


# EVAL WITH NO SUBSET ##
# This is how you create a simple task (like hellaswag) which has one single subset
# attached to it, and one evaluation possible.
task_en2my = LightevalTaskConfig(
    name="mt:en2my",
    suite=["custom"],
    prompt_function=prompt_fn,  
    hf_repo="your-hf-id/flores", 
    hf_subset="default",
    hf_avail_splits=["test", "validation"], 
    evaluation_splits=["test"], 
    few_shots_split=["validation"], 
    few_shots_select="random_sampling",
    metric=[chrf_sample],
    generation_size=128, # the maximum number of tokens allowed for a generative evaluation. If your evaluation is a log likelihood evaluation (multi-choice), this value should be -1
    stop_sequence=["\n"], 
    output_regex=None, # A regex string that will be used to filter your generation. (Generative metrics will only select tokens that are between the first and the second sequence matched by the regex. For example, for a regex matching \n and a generation \nModel generation output\nSome other text the metric will only be fed with Model generation output)
    trust_dataset=True,
)

task_my2en = LightevalTaskConfig(
    name="mt:my2en",
    suite=["custom"],
    prompt_function=prompt_fn,  
    hf_repo="your-hf-id/flores", 
    hf_subset="default",
    hf_avail_splits=["test", "validation"], 
    evaluation_splits=["test"], 
    few_shots_split=["validation"], 
    few_shots_select="random_sampling",
    metric=[chrf_sample],
    generation_size=128, # the maximum number of tokens allowed for a generative evaluation. If your evaluation is a log likelihood evaluation (multi-choice), this value should be -1
    stop_sequence=["\n"], 
    output_regex=None, # A regex string that will be used to filter your generation. (Generative metrics will only select tokens that are between the first and the second sequence matched by the regex. For example, for a regex matching \n and a generation \nModel generation output\nSome other text the metric will only be fed with Model generation output)
    trust_dataset=True,
)

task_en2si = LightevalTaskConfig(
    name="mt:en2si",
    suite=["custom"],
    prompt_function=prompt_fn,  
    hf_repo="your-hf-id/flores", 
    hf_subset="default",
    hf_avail_splits=["test", "validation"], 
    evaluation_splits=["test"], 
    few_shots_split=["validation"], 
    few_shots_select="random_sampling",
    metric=[chrf_sample],
    generation_size=128, # the maximum number of tokens allowed for a generative evaluation. If your evaluation is a log likelihood evaluation (multi-choice), this value should be -1
    stop_sequence=["\n"], 
    output_regex=None, # A regex string that will be used to filter your generation. (Generative metrics will only select tokens that are between the first and the second sequence matched by the regex. For example, for a regex matching \n and a generation \nModel generation output\nSome other text the metric will only be fed with Model generation output)
    trust_dataset=True,
)

task_si2en = LightevalTaskConfig(
    name="mt:si2en",
    suite=["custom"],
    prompt_function=prompt_fn,  
    hf_repo="your-hf-id/flores", 
    hf_subset="default",
    hf_avail_splits=["test", "validation"], 
    evaluation_splits=["test"], 
    few_shots_split=["validation"], 
    few_shots_select="random_sampling",
    metric=[chrf_sample],
    generation_size=128, # the maximum number of tokens allowed for a generative evaluation. If your evaluation is a log likelihood evaluation (multi-choice), this value should be -1
    stop_sequence=["\n"], 
    output_regex=None, # A regex string that will be used to filter your generation. (Generative metrics will only select tokens that are between the first and the second sequence matched by the regex. For example, for a regex matching \n and a generation \nModel generation output\nSome other text the metric will only be fed with Model generation output)
    trust_dataset=True,
)

task_en2te = LightevalTaskConfig(
    name="mt:en2te",
    suite=["custom"],
    prompt_function=prompt_fn,  
    hf_repo="your-hf-id/flores", 
    hf_subset="default",
    hf_avail_splits=["test", "validation"], 
    evaluation_splits=["test"], 
    few_shots_split=["validation"], 
    few_shots_select="random_sampling",
    metric=[chrf_sample],
    generation_size=128, # the maximum number of tokens allowed for a generative evaluation. If your evaluation is a log likelihood evaluation (multi-choice), this value should be -1
    stop_sequence=["\n"], 
    output_regex=None, # A regex string that will be used to filter your generation. (Generative metrics will only select tokens that are between the first and the second sequence matched by the regex. For example, for a regex matching \n and a generation \nModel generation output\nSome other text the metric will only be fed with Model generation output)
    trust_dataset=True,
)

task_te2en = LightevalTaskConfig(
    name="mt:te2en",
    suite=["custom"],
    prompt_function=prompt_fn,  
    hf_repo="your-hf-id/flores", 
    hf_subset="default",
    hf_avail_splits=["test", "validation"], 
    evaluation_splits=["test"], 
    few_shots_split=["validation"], 
    few_shots_select="random_sampling",
    metric=[chrf_sample],
    generation_size=128, # the maximum number of tokens allowed for a generative evaluation. If your evaluation is a log likelihood evaluation (multi-choice), this value should be -1
    stop_sequence=["\n"], 
    output_regex=None, # A regex string that will be used to filter your generation. (Generative metrics will only select tokens that are between the first and the second sequence matched by the regex. For example, for a regex matching \n and a generation \nModel generation output\nSome other text the metric will only be fed with Model generation output)
    trust_dataset=True,
)

task_en2ta = LightevalTaskConfig(
    name="mt:en2ta",
    suite=["custom"],
    prompt_function=prompt_fn,  
    hf_repo="your-hf-id/flores", 
    hf_subset="default",
    hf_avail_splits=["test", "validation"], 
    evaluation_splits=["test"], 
    few_shots_split=["validation"], 
    few_shots_select="random_sampling",
    metric=[chrf_sample],
    generation_size=128, # the maximum number of tokens allowed for a generative evaluation. If your evaluation is a log likelihood evaluation (multi-choice), this value should be -1
    stop_sequence=["\n"], 
    output_regex=None, # A regex string that will be used to filter your generation. (Generative metrics will only select tokens that are between the first and the second sequence matched by the regex. For example, for a regex matching \n and a generation \nModel generation output\nSome other text the metric will only be fed with Model generation output)
    trust_dataset=True,
)

task_ta2en = LightevalTaskConfig(
    name="mt:ta2en",
    suite=["custom"],
    prompt_function=prompt_fn,  
    hf_repo="your-hf-id/flores", 
    hf_subset="default",
    hf_avail_splits=["test", "validation"], 
    evaluation_splits=["test"], 
    few_shots_split=["validation"], 
    few_shots_select="random_sampling",
    metric=[chrf_sample],
    generation_size=128, # the maximum number of tokens allowed for a generative evaluation. If your evaluation is a log likelihood evaluation (multi-choice), this value should be -1
    stop_sequence=["\n"], 
    output_regex=None, # A regex string that will be used to filter your generation. (Generative metrics will only select tokens that are between the first and the second sequence matched by the regex. For example, for a regex matching \n and a generation \nModel generation output\nSome other text the metric will only be fed with Model generation output)
    trust_dataset=True,
)

task_en2am = LightevalTaskConfig(
    name="mt:en2am",
    suite=["custom"],
    prompt_function=prompt_fn,  
    hf_repo="your-hf-id/flores", 
    hf_subset="default",
    hf_avail_splits=["test", "validation"], 
    evaluation_splits=["test"], 
    few_shots_split=["validation"], 
    few_shots_select="random_sampling",
    metric=[chrf_sample],
    generation_size=128, # the maximum number of tokens allowed for a generative evaluation. If your evaluation is a log likelihood evaluation (multi-choice), this value should be -1
    stop_sequence=["\n"], 
    output_regex=None, # A regex string that will be used to filter your generation. (Generative metrics will only select tokens that are between the first and the second sequence matched by the regex. For example, for a regex matching \n and a generation \nModel generation output\nSome other text the metric will only be fed with Model generation output)
    trust_dataset=True,
)

task_am2en = LightevalTaskConfig(
    name="mt:am2en",
    suite=["custom"],
    prompt_function=prompt_fn,  
    hf_repo="your-hf-id/flores", 
    hf_subset="default",
    hf_avail_splits=["test", "validation"], 
    evaluation_splits=["test"], 
    few_shots_split=["validation"], 
    few_shots_select="random_sampling",
    metric=[chrf_sample],
    generation_size=128, # the maximum number of tokens allowed for a generative evaluation. If your evaluation is a log likelihood evaluation (multi-choice), this value should be -1
    stop_sequence=["\n"], 
    output_regex=None, # A regex string that will be used to filter your generation. (Generative metrics will only select tokens that are between the first and the second sequence matched by the regex. For example, for a regex matching \n and a generation \nModel generation output\nSome other text the metric will only be fed with Model generation output)
    trust_dataset=True,
)

task_en2bn = LightevalTaskConfig(
    name="mt:en2bn",
    suite=["custom"],
    prompt_function=prompt_fn,  
    hf_repo="your-hf-id/flores", 
    hf_subset="default",
    hf_avail_splits=["test", "validation"], 
    evaluation_splits=["test"], 
    few_shots_split=["validation"], 
    few_shots_select="random_sampling",
    metric=[chrf_sample],
    generation_size=128, # the maximum number of tokens allowed for a generative evaluation. If your evaluation is a log likelihood evaluation (multi-choice), this value should be -1
    stop_sequence=["\n"], 
    output_regex=None, # A regex string that will be used to filter your generation. (Generative metrics will only select tokens that are between the first and the second sequence matched by the regex. For example, for a regex matching \n and a generation \nModel generation output\nSome other text the metric will only be fed with Model generation output)
    trust_dataset=True,
)

task_bn2en = LightevalTaskConfig(
    name="mt:bn2en",
    suite=["custom"],
    prompt_function=prompt_fn,  
    hf_repo="your-hf-id/flores", 
    hf_subset="default",
    hf_avail_splits=["test", "validation"], 
    evaluation_splits=["test"], 
    few_shots_split=["validation"], 
    few_shots_select="random_sampling",
    metric=[chrf_sample],
    generation_size=128, # the maximum number of tokens allowed for a generative evaluation. If your evaluation is a log likelihood evaluation (multi-choice), this value should be -1
    stop_sequence=["\n"], 
    output_regex=None, # A regex string that will be used to filter your generation. (Generative metrics will only select tokens that are between the first and the second sequence matched by the regex. For example, for a regex matching \n and a generation \nModel generation output\nSome other text the metric will only be fed with Model generation output)
    trust_dataset=True,
)

task_en2gu = LightevalTaskConfig(
    name="mt:en2gu",
    suite=["custom"],
    prompt_function=prompt_fn,  
    hf_repo="your-hf-id/flores", 
    hf_subset="default",
    hf_avail_splits=["test", "validation"], 
    evaluation_splits=["test"], 
    few_shots_split=["validation"], 
    few_shots_select="random_sampling",
    metric=[chrf_sample],
    generation_size=128, # the maximum number of tokens allowed for a generative evaluation. If your evaluation is a log likelihood evaluation (multi-choice), this value should be -1
    stop_sequence=["\n"], 
    output_regex=None, # A regex string that will be used to filter your generation. (Generative metrics will only select tokens that are between the first and the second sequence matched by the regex. For example, for a regex matching \n and a generation \nModel generation output\nSome other text the metric will only be fed with Model generation output)
    trust_dataset=True,
)

task_gu2en = LightevalTaskConfig(
    name="mt:gu2en",
    suite=["custom"],
    prompt_function=prompt_fn,  
    hf_repo="your-hf-id/flores", 
    hf_subset="default",
    hf_avail_splits=["test", "validation"], 
    evaluation_splits=["test"], 
    few_shots_split=["validation"], 
    few_shots_select="random_sampling",
    metric=[chrf_sample],
    generation_size=128, # the maximum number of tokens allowed for a generative evaluation. If your evaluation is a log likelihood evaluation (multi-choice), this value should be -1
    stop_sequence=["\n"], 
    output_regex=None, # A regex string that will be used to filter your generation. (Generative metrics will only select tokens that are between the first and the second sequence matched by the regex. For example, for a regex matching \n and a generation \nModel generation output\nSome other text the metric will only be fed with Model generation output)
    trust_dataset=True,
)

# STORE YOUR EVALS
TASKS_TABLE = [task_en2my, task_my2en, task_en2si, task_si2en, task_en2te, task_te2en,
               task_en2ta, task_ta2en, task_en2am, task_am2en, task_en2bn, task_bn2en,
               task_en2gu, task_gu2en]


# MODULE LOGIC
# You should not need to touch this
# Convert to dict for lighteval
if __name__ == "__main__":
    print(t.name for t in TASKS_TABLE)
    print(len(TASKS_TABLE))
