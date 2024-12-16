# See https://github.com/huggingface/lighteval/wiki/Adding-a-Custom-Task

import os

import numpy as np
from aenum import extend_enum
from multi_lingual_rouge_score import multi_lingual_rouge as rouge_scorer

from lighteval.metrics.metrics import Metrics, SampleLevelMetric
from lighteval.metrics.metrics_sample import ROUGE
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
    summary = line["summary"]
    text = line["text"]
    lang_code = task_name.split(":")[1]
    if lang_code == "my":
        return Doc(
            task_name=task_name,
            query=f"အောက်ပါစာသားကို မြန်မာဘာသာဖြင့် အကျဉ်းချုပ်ရေးပါ။ ဆောင်းပါး: {text} အကျဉ်းချုပ်:",
            gold_index=0,
            choices=[str(summary)],
            specific={"text": text},
        )
    elif lang_code == "si":
        return Doc(
            task_name=task_name,
            query=f"පහත පාඨයේ සාරාංශය සිංහලෙන් ලියන්න. ලිපිය: {text} සාරාංශය:",
            gold_index=0,
            choices=[str(summary)],
            specific={"text": text},
        )
    elif lang_code == "te":
        return Doc(
            task_name=task_name,
            query=f"క్రింది వచనం యొక్క సారాంశం తెలుగులో రాయండి. వ్యాసం: {text} సారాంశం:",
            gold_index=0,
            choices=[str(summary)],
            specific={"text": text},
        )
    elif lang_code == "ta":
        return Doc(
            task_name=task_name,
            query=f"கீழே உள்ள உரையை தமிழில் சுருக்கமாக எழுதுங்கள்: {text} சுருக்கம்:",
            gold_index=0,
            choices=[str(summary)],
            specific={"text": text},
        )
    elif lang_code == "bn":
        return Doc(
            task_name=task_name,
            query=f"নিম্নলিখিত লেখাটি বাংলায় সংক্ষেপে লিখুন।: {text} সংক্ষিপ্তসার:",
            gold_index=0,
            choices=[str(summary)],
            specific={"text": text},
        )
    elif lang_code == "am":
        return Doc(
            task_name=task_name,
            query=f"የታችኛው ጽሁፍን በአማርኛ አጭር በማድረግ አሳትረኝ።: {text} አጭር መግለጫ:",
            gold_index=0,
            choices=[str(summary)],
            specific={"text": text},
        )
    elif lang_code == "gu":
        return Doc(
            task_name=task_name,
            query=f"નીચે આપેલા લખાણને ગુજરાતીમાં સંક્ષિપ્ત લખો.: {text} સંક્ષેપ:",
            gold_index=0,
            choices=[str(summary)],
            specific={"text": text},
        )
    elif lang_code == "en":
        return Doc(
            task_name=task_name,
            query=f"Summarize the following text in English: {text} Summary:",
            gold_index=0,
            choices=[str(summary)],
            specific={"text": text},
        )
    else:
        raise NotImplementedError


# CUSTOM METRIC IF NEEDED
class MultilingualROUGE(ROUGE):
    ALLOWED_ROUGE_METHODS = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def _rouge_score(self, golds: list[str], preds: list[str]):
        scores = {m: [] for m in self.methods}
        for pred in preds:
            for gold in golds:
                cur_scores = self.scorer.score(gold, pred)
                for method in self.methods:
                    scores[method].append(cur_scores[method].fmeasure)
        return {method: self.aggregation_function(scores[method]) for method in self.methods}


rougeL_multilingual = SampleLevelMetric(
    metric_name="rougeL_multilingual",
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.SUMMARIZATION,
    sample_level_fn=MultilingualROUGE("rougeL").compute, # how to compute score for one sample
    corpus_level_fn=np.mean, # aggregation
    higher_is_better=True,
)
extend_enum(Metrics, "rougeL_multilingual", rougeL_multilingual)


# EVAL WITH NO SUBSET ##
# This is how you create a simple task (like hellaswag) which has one single subset
# attached to it, and one evaluation possible.
task_my = LightevalTaskConfig(
    name="sum:my",
    suite=["custom"],
    prompt_function=prompt_fn,  
    hf_repo="your-hf-id/sum-my", 
    hf_subset="default",
    hf_avail_splits=["test"], 
    evaluation_splits=["test"], 
    few_shots_split=None, 
    few_shots_select=None,
    metric=[rougeL_multilingual],
    generation_size=128, # the maximum number of tokens allowed for a generative evaluation. If your evaluation is a log likelihood evaluation (multi-choice), this value should be -1
    stop_sequence=["\n"], 
    output_regex=None, # A regex string that will be used to filter your generation. (Generative metrics will only select tokens that are between the first and the second sequence matched by the regex. For example, for a regex matching \n and a generation \nModel generation output\nSome other text the metric will only be fed with Model generation output)
    trust_dataset=True,
)

task_te = LightevalTaskConfig(
    name="sum:te",
    suite=["custom"],
    prompt_function=prompt_fn,  
    hf_repo="your-hf-id/sum-te", 
    hf_subset="default",
    hf_avail_splits=["test"], 
    evaluation_splits=["test"], 
    few_shots_split=None, 
    few_shots_select=None,
    metric=[rougeL_multilingual],
    generation_size=128, # the maximum number of tokens allowed for a generative evaluation. If your evaluation is a log likelihood evaluation (multi-choice), this value should be -1
    stop_sequence=["\n"], 
    output_regex=None, # A regex string that will be used to filter your generation. (Generative metrics will only select tokens that are between the first and the second sequence matched by the regex. For example, for a regex matching \n and a generation \nModel generation output\nSome other text the metric will only be fed with Model generation output)
    trust_dataset=True,
)

task_si = LightevalTaskConfig(
    name="sum:si",
    suite=["custom"],
    prompt_function=prompt_fn,  
    hf_repo="your-hf-id/sum-si", 
    hf_subset="default",
    hf_avail_splits=["test"], 
    evaluation_splits=["test"], 
    few_shots_split=None, 
    few_shots_select=None,
    metric=[rougeL_multilingual],
    generation_size=128, # the maximum number of tokens allowed for a generative evaluation. If your evaluation is a log likelihood evaluation (multi-choice), this value should be -1
    stop_sequence=["\n"], 
    output_regex=None, # A regex string that will be used to filter your generation. (Generative metrics will only select tokens that are between the first and the second sequence matched by the regex. For example, for a regex matching \n and a generation \nModel generation output\nSome other text the metric will only be fed with Model generation output)
    trust_dataset=True,
)

task_ta = LightevalTaskConfig(
    name="sum:ta",
    suite=["custom"],
    prompt_function=prompt_fn,  
    hf_repo="your-hf-id/sum-ta", 
    hf_subset="default",
    hf_avail_splits=["test"], 
    evaluation_splits=["test"], 
    few_shots_split=None, 
    few_shots_select=None,
    metric=[rougeL_multilingual],
    generation_size=128, # the maximum number of tokens allowed for a generative evaluation. If your evaluation is a log likelihood evaluation (multi-choice), this value should be -1
    stop_sequence=["\n"], 
    output_regex=None, # A regex string that will be used to filter your generation. (Generative metrics will only select tokens that are between the first and the second sequence matched by the regex. For example, for a regex matching \n and a generation \nModel generation output\nSome other text the metric will only be fed with Model generation output)
    trust_dataset=True,
)

task_bn = LightevalTaskConfig(
    name="sum:bn",
    suite=["custom"],
    prompt_function=prompt_fn,
    hf_repo="your-hf-id/sum-bn", 
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    metric=[rougeL_multilingual],
    generation_size=128,  # the maximum number of tokens allowed for a generative evaluation. If your evaluation is a log likelihood evaluation (multi-choice), this value should be -1
    stop_sequence=["\n"],  
    output_regex=None,  # A regex string that will be used to filter your generation. (Generative metrics will only select tokens that are between the first and the second sequence matched by the regex. For example, for a regex matching \n and a generation \nModel generation output\nSome other text the metric will only be fed with Model generation output)
    trust_dataset=True,
)

task_am = LightevalTaskConfig(
    name="sum:am",
    suite=["custom"],
    prompt_function=prompt_fn,
    hf_repo="your-hf-id/sum-am", 
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    metric=[rougeL_multilingual],
    generation_size=128,  # the maximum number of tokens allowed for a generative evaluation. If your evaluation is a log likelihood evaluation (multi-choice), this value should be -1
    stop_sequence=["\n"],  
    output_regex=None,  # A regex string that will be used to filter your generation. (Generative metrics will only select tokens that are between the first and the second sequence matched by the regex. For example, for a regex matching \n and a generation \nModel generation output\nSome other text the metric will only be fed with Model generation output)
    trust_dataset=True,
)

task_gu = LightevalTaskConfig(
    name="sum:gu",
    suite=["custom"],
    prompt_function=prompt_fn,
    hf_repo="your-hf-id/sum-gu", 
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    metric=[rougeL_multilingual],
    generation_size=128,  # the maximum number of tokens allowed for a generative evaluation. If your evaluation is a log likelihood evaluation (multi-choice), this value should be -1
    stop_sequence=["\n"],  
    output_regex=None,  # A regex string that will be used to filter your generation. (Generative metrics will only select tokens that are between the first and the second sequence matched by the regex. For example, for a regex matching \n and a generation \nModel generation output\nSome other text the metric will only be fed with Model generation output)
    trust_dataset=True,
)

task_en = LightevalTaskConfig(
    name="sum:en",
    suite=["custom"],
    prompt_function=prompt_fn,  
    hf_repo="your-hf-id/sum-en", 
    hf_subset="default",
    hf_avail_splits=["test"], 
    evaluation_splits=["test"], 
    few_shots_split=None, 
    few_shots_select=None,
    metric=[rougeL_multilingual],
    generation_size=128, # the maximum number of tokens allowed for a generative evaluation. If your evaluation is a log likelihood evaluation (multi-choice), this value should be -1
    stop_sequence=["\n"], 
    output_regex=None, # A regex string that will be used to filter your generation. (Generative metrics will only select tokens that are between the first and the second sequence matched by the regex. For example, for a regex matching \n and a generation \nModel generation output\nSome other text the metric will only be fed with Model generation output)
    trust_dataset=True,
)

# STORE YOUR EVALS
TASKS_TABLE = [task_my, task_te, task_si, task_ta, task_bn, task_am, task_gu, task_en]


# MODULE LOGIC
# You should not need to touch this
# Convert to dict for lighteval
if __name__ == "__main__":
    print(t.name for t in TASKS_TABLE)
    print(len(TASKS_TABLE))
