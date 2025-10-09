import os

# Set custom cache directories
os.environ["HF_HOME"] = "/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
os.environ["TORCH_HOME"] = "/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
os.environ["NEMO_CACHE_DIR"] = "/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Hugging Face access token
access_token = None

# model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
# model_id = "meta-llama/Llama-3.1-70B-Instruct"
# model_id = "meta-llama/Llama-3.1-8B-Instruct"
pipe = pipeline("text-generation", model=model_id, tokenizer=model_id, token=access_token)

# text = "<|2|> swear <|9|> <|9|> to <|10|> <|10|> me <|12|> <|14|> walton <|20|> <|25|> that <|26|> <|28|> he <|29|> <|30|> shall <|33|> <|34|> not <|35|> <|37|> escape <|42|> <|53|> that <|54|> <|56|> you <|57|> <|59|> will <|60|> <|62|> seek <|65|> <|65|> him <|66|> <|68|> and <|69|> <|70|> satisfy <|77|> <|79|> my <|80|> <|81|> vengeance <|87|> <|91|> in <|92|> <|94|> his <|95|> <|97|> death <|98|> <|111|> and <|112|> <|115|> do <|116|> <|116|> i <|118|> <|119|> dare <|122|> <|123|> to <|124|> <|125|> ask <|128|> <|128|> of <|129|> <|130|> you <|131|> <|133|> to <|134|> <|134|> undertake <|140|> <|141|> my <|142|> <|143|> pilgrimage <|151|> <|153|> to <|154|> <|154|> endure <|159|> <|159|> the <|160|> <|160|> hardships <|167|> <|168|> that <|169|> <|169|> i <|171|> <|171|> have <|172|> <|173|> undergone <|179|> <|190|> no <|191|>"
# text = "<|4|> was <|5|> <|8|> she <|9|> <|10|> a <|11|> <|12|> very <|13|> <|14|> charming <|19|> <|20|> woman <|23|> <|25|> was <|26|> <|29|> she <|30|> <|31|> handsome <|36|> <|38|> was <|39|> <|42|> there <|43|> <|44|> any <|45|> <|46|> picture <|50|> <|50|> of <|51|> <|51|> her <|52|> <|53|> in <|54|> <|54|> the <|55|> <|55|> abbey <|62|>"
# text = "<|4|> for <|5|> <|8|> once <|11|> <|13|> he <|14|> <|15|> was <|16|> <|17|> grateful <|21|> <|22|> to <|23|> <|23|> the <|24|> <|24|> forest <|28|> <|30|> because <|31|> <|32|> it <|33|> <|33|> had <|34|> <|35|> forbidden <|41|> <|41|> him <|42|> <|42|> to <|43|> <|44|> glance <|47|> <|48|> perpetually <|55|> <|55|> back <|57|> <|58|> at <|59|> <|59|> that <|60|> <|61|> dismal <|66|> <|67|> and <|68|> <|69|> pensive <|74|> <|74|> picture <|79|> <|83|> then <|86|> <|87|> he <|88|> <|89|> became <|90|> <|92|> aware <|96|> <|97|> of <|98|> <|99|> twigs <|104|> <|106|> hastily <|111|> <|111|> lopped <|116|> <|117|> off <|118|> <|128|> of <|129|> <|131|> bushes <|137|> <|137|> bent <|140|> <|141|> and <|142|> <|143|> torn <|147|> <|150|> of <|151|> <|152|> the <|153|> <|154|> uncovering <|162|> <|169|> through <|170|> <|174|> these <|175|> <|177|> careless <|182|> <|183|> means <|188|> <|189|> of <|190|> <|191|> an <|192|> <|193|> old <|194|> <|196|> path <|200|>"
# text = "<|2|> it's <|5|> <|5|> natural <|10|> <|11|> enough <|12|> <|14|> he <|15|> <|16|> should <|17|> <|18|> be <|19|> <|21|> here <|22|> <|30|> bobby <|37|> <|38|> agreed <|42|> <|42|> indifferently <|51|> <|53|> they <|54|> <|57|> walked <|60|> <|61|> slowly <|66|> <|67|> back <|69|> <|70|> to <|71|> <|71|> the <|73|> <|73|> house <|74|> <|84|> graham <|90|> <|91|> made <|92|> <|93|> it <|94|> <|94|> plain <|98|> <|99|> that <|100|> <|101|> his <|102|> <|103|> mind <|104|> <|106|> was <|107|> <|108|> far <|109|> <|112|> from <|113|> <|114|> the <|115|> <|115|> sad <|119|> <|120|> business <|121|> <|124|> ahead <|129|>"
text = "<|2|> he <|3|> <|5|> would <|6|> <|7|> have <|8|> <|9|> interested <|15|> <|16|> himself <|17|> <|22|> somewhat <|28|> <|29|> more <|30|> <|32|> about <|33|> <|35|> it <|36|> <|46|> still <|47|> <|52|> said <|53|> <|54|> chateau <|61|> <|62|> renaud <|68|>"

prompt = f"""
#### AI Assistant Information
- Response Requirements: Segment a given text into turns
- Input text: The <|integer|> means the timestamps of a word. Each word has a start and an end timestamp. The integer is the frame index with a frame rate of 80 ms.
- Parsing tips: From one turn to the next, usually there is a duration gap.
- Prohibited Content: The AI assistant must not provide illegal or harmful information and should refuse to answer politically sensitive questions.
- No empty turns: Do not include empty turns.

#### Turn Segmentation Task
- Number of turns: Generate a number of turns you think is appropriate
- Exact match: The output should be exactly the same as the Input_without_timestamps, except the turns are added.

#### Output Format
"turn_1": <first_turn>
"turn_2": <second_turn>
"turn_3": <third_turn>
"turn_4": <forth_turn>

#### Example
This is an example of how the output should look like given the input text:

## Input:
<|3|> he <|4|> <|6|> said <|7|> <|15|> how <|16|> <|19|> happy <|20|> <|24|> you <|25|> <|27|> will <|28|> <|30|> be <|31|> <|54|> i <|59|> <|59|> will <|60|> <|63|> do <|64|> <|66|> my <|67|> <|70|> best <|71|> <|83|> said <|84|> <|86|> the <|87|> <|88|> inn <|91|> <|92|> keeper <|96|> <|97|> of <|98|> <|99|> the <|100|> <|100|> pont <|104|> <|105|> du <|106|> <|108|> gard <|112|> <|117|> shutting <|123|> <|123|> up <|124|> <|126|> his <|127|> <|128|> knife <|132|> <|155|> well <|156|> <|160|> then <|163|> <|166|> we <|167|> <|168|> will <|169|> <|171|> go <|172|> <|173|> into <|174|> <|175|> paris <|180|>

## Input_without_timestamps:
he said how happy you will be i will do my best said the inn keeper of the pont du gard shutting up his knife well then we will go into paris

## Output:
turn_1: he said how happy you will be
turn_2: i will do my best said the inn keeper of the pont du gard shutting up his knife
turn_3: well then we will go into paris
## End of Output

Now parse the following text into turns:
{text}

#### Output
"""

out = pipe(prompt, max_new_tokens=200, do_sample=False)
generated_text = out[0]["generated_text"]

print("Generated text:")
print(generated_text)
