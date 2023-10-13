import openai
import time
import random
import csv
from copy import deepcopy
import re
import pandas as pd
from pip._vendor.webencodings.tests import test_all_labels
openai.api_key="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"
#openai.api_key="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"

def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.error.RateLimitError, openai.error.Timeout, openai.error.APIConnectionError, openai.error.APIError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def paraphrase(ori_examples, num_paraphrases = 2, dump_file = None, cache_file = None):
    
    para_examples = []
    
    if cache_file is None:
    
        with open(dump_file,"a",newline="\n") as f:
            csv_writer=csv.writer(f)
            csv_writer.writerows([("GUID", "Original Example", "ChatGPT's Paraphrase", "Label")])
        
        for ori_example in ori_examples:
            
            start = time.time()
            
            batch_para_examples = []
            
            if num_paraphrases == 1:
                instruction = "Please paraphrase the sentence below:\n"
            else:
                instruction = "Please generate %d paraphrases of the sentence below:\n"%num_paraphrases
               
            if num_paraphrases == 1:
                paraphrases = []   
                response = completions_with_backoff(
                    model="gpt-3.5-turbo", 
                    messages=[{"role": "user", "content": instruction + ori_example.text_a}]
                )
                
                content = response.choices[0].message.content
                paraphrases.append(content)
            
            else:
                #paraphrases = []
                contents = []
                content=""
                while not content.split('\n')[-1].startswith(str(num_paraphrases)):
                    messages = []
                    messages.append({"role": "user", "content": instruction + ori_example.text_a})
                    if len(contents) > 0:
                        messages.append({"role": "assistant", "content": contents[-1]})
                        messages.append({"role": "user", "content": "continue"})
                    
                    response = completions_with_backoff(
                        model="gpt-3.5-turbo", 
                        messages=messages, 
                        max_tokens=1920
                    )
                
                    content = response.choices[0].message.content
                    contents.append(content)
                    
                '''   
                for content in contents:
                    rows = content.split('\n')
                    if len(paraphrases) > 0:
                        mo_0 = re.match(r"([0-9]+).*", paraphrases[-1])
                        print("paraphrases[-1]:",paraphrases[-1])
                        num_0 = None
                        if mo_0:
                            num_0 = mo_0.group(1)
                        
                        mo_1 = re.match(r"([0-9]+).*", rows[0])
                        print("rows[0]:",rows[0])
                        num_1 = None
                        if mo_1:
                            num_1 = mo_1.group(1)
                        if num_0 is not None and num_1 is not None and num_0 == num_1:
                            paraphrases = paraphrases[:-1]
                    
                    paraphrases.extend(rows)
                '''
                paraphrases = (''.join(contents)).split('\n')
                
            for paraphrase in paraphrases:
                if len(paraphrase.strip()) > 0:
                    para_example = deepcopy(ori_example)
                    print("label:", para_example.label)
                    print("original sentence:", para_example.text_a)
                    para_example.text_b = para_example.text_a
                    mo = re.match(r"[0-9]+\.\s+(.*)", paraphrase)
                    if mo:
                        paraphrase = mo.group(1)
                    para_example.text_a = paraphrase
                    print("chatgpt's paraphrase:", para_example.text_a)
                    print('-'*80)
                    batch_para_examples.append(para_example)
                
            with open(dump_file,"a",newline="\n") as f:
                csv_writer=csv.writer(f)
                csv_writer.writerows([(para_example.guid, para_example.text_b, para_example.text_a, para_example.label) for para_example in batch_para_examples])  
            
            para_examples.extend(batch_para_examples)
            
            unique_examples={}
            for para_example in batch_para_examples:
                if para_example not in unique_examples:
                    unique_examples[para_example] = 0
                unique_examples[para_example] +=1
            print("# of unique examples:", len(unique_examples))
            
            end = time.time()
            print("elapsed time:", end - start)
    else:
        
        df = pd.read_csv(cache_file)
        df = df[df['GUID'].isin([ori_example.guid for ori_example in ori_examples])]
        
        for ori_example in ori_examples:
            example_df = df[df['GUID'] == ori_example.guid]   
            paraphrases = example_df["ChatGPT's Paraphrase"].tolist()
            
            '''
            print("-"*80)
            print(ori_example.text_a)
            print(paraphrases)
            print("-"*80)
            '''
            
            paraphrases = random.sample(paraphrases, k=num_paraphrases)
            for paraphrase in paraphrases:
                para_example = deepcopy(ori_example)
                para_example.text_b = para_example.text_a
                para_example.text_a = paraphrase
                para_examples.append(para_example)
                        
    return para_examples


def predict(train_df, test_df, unseen_label, known_cls_ratio, seed, num_shots = 2, mini_batch_size = 4):
    known_label_list = train_df.category.unique()
    degree_adverb_map = {"0.25": "", "0.5": "kind of ", "0.75": "very "}
    
    #sampled_train_df = train_df.groupby('category').max().reset_index()
    random.seed(seed)
    fn = lambda obj: obj.loc[random.sample(list(obj.index), num_shots),:]
    sampled_train_df = train_df.groupby('category', as_index=False).apply(fn).sample(frac = 1)
    
    
    context = "There are {} categories. They are {}. \nHere are some examles:\n".format(len(train_df.category.unique()), ', '.join(list(train_df.category.unique())))
    
    for index, train_example in sampled_train_df.iterrows():
        context += ('"{}" Its category is {}.\n'.format(train_example.text, train_example.category))
    
    print("context:", context)
    
    accuracy = 0
    NUM_TEST_EXAMPLES = len(test_df)
    
    all_labels = []
    all_predictions = []
    
    mb_indexes = []
    mb_questions = []
    mb_labels = []
    for index, (_, test_example) in enumerate(random.sample(list(test_df.iterrows()),NUM_TEST_EXAMPLES)):
        
        
        mb_indexes.append(index)
        mb_questions.append('"{}"'.format(test_example.text))
        mb_labels.append(test_example.category)
        
        if index % mini_batch_size == mini_batch_size -1 or index == NUM_TEST_EXAMPLES - 1:
            
            
            combined_question = 'What {} the {} for {}? Return the "<UNK>" category if it is {}uncertain. {}'.format(
                "is" if mini_batch_size == 1 else "are", 
                "category" if mini_batch_size == 1 else "categories", 
                " and ".join(mb_questions),
                degree_adverb_map[known_cls_ratio],
                "" if mini_batch_size == 1 else ' Display a table with "Question" and "Category" columns, and only include the first two words of the "Question". Separate the two columns with a vertical bar.')
    
            prompt = context + combined_question
            
            #print("prompt:", prompt)
            
        
            completions = completions_with_backoff(
                model="gpt-3.5-turbo", 
                messages=[{"role": "user", "content": prompt}]
            )
            '''
            with open('dump.txt','a') as f:
                f.write(completions["choices"][0]["message"]["content"])
            '''
            #print("completions:", completions)
            content = completions["choices"][0]["message"]["content"]
            print("content:", content)
            if mini_batch_size == 1:
                mb_answers = [content]
            else:
                mb_answers = re.compile("\n+").split(content)
                sep_idx = 0
                for i, line in enumerate(mb_answers):
                    mo = re.search(r"(\-)+(\s)*\|(\s)*(\-)+", line)
                    if mo:
                        sep_idx = i
                        break
                mb_answers = mb_answers[sep_idx+1:]
    
            for mb_index, mb_question, mb_answer, mb_label in zip(mb_indexes, mb_questions, mb_answers, mb_labels):
                
                print("index:", mb_index)
                print("question:", mb_question)
                print("answer:", mb_answer)
                print("label:", mb_label)
                
                all_labels.append(mb_label)
                
                if mini_batch_size == 1:
                    mo = re.match(r'.*category .*(is|would be|could be|can be|could fall under) (.+)\.', mb_answer)
                    prediction_group_index = 2
                else:
                    mo = re.match(r'.+\|(.+)', mb_answer)
                    prediction_group_index = 1
                if mo:
                    prediction_group = mo.group(prediction_group_index)
                    mb_prediction = unseen_label
                    for label in known_label_list:
                        if label in prediction_group:
                            mb_prediction = label
                            break
                    
                    accuracy += (mb_label == mb_prediction)
                    print("predction:", mb_prediction)
                    
                    all_predictions.append(mb_prediction)
                else:
                    print("Error matching prediction.")
                    all_predictions.append(unseen_label)
                print('-'*80)
                    
            mb_indexes = []
            mb_questions = []
            mb_labels = []
            mb_answers = []
    
    print("accuracy:", accuracy/NUM_TEST_EXAMPLES)
    
    return all_labels, all_predictions
