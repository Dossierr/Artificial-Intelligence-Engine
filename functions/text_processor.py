import boto3
import redis
import environ
from langchain.cache import RedisCache
from langchain.memory import RedisChatMessageHistory
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.globals import set_llm_cache
from langchain.llms.bedrock import Bedrock
from langchain.callbacks import get_openai_callback
import tiktoken



#loading ENV variables
env = environ.Env()
environ.Env.read_env()

#Connecting cache for LLM
r = redis.Redis(
  host='redis-15281.c300.eu-central-1-1.ec2.cloud.redislabs.com',
  port=15281,
  password=env('REDIS_PASSWORD'),
  decode_responses=True)

#Uses Redis as cache for frequent queries witha time to live
set_llm_cache(RedisCache(r, ttl=60*60))

#Define credentials
access_key_id = env('BEDROCK_AWS_ACCESS_KEY_ID')
secret_access_key = env('BEDROCK_AWS_SECRET_KEY')

#Log into bedrock as a client
client_boto3 = boto3.client(service_name="bedrock-runtime", 
                       aws_access_key_id=access_key_id, 
                       aws_secret_access_key=secret_access_key)

#Defining the Model we want to use
modelId = "amazon.titan-text-express-v1"         
titan_llm = Bedrock(model_id=modelId, client=client_boto3, streaming=False)
titan_llm.model_kwargs = {'temperature': 0.1}

#Setting memory for conversation
memory = ConversationBufferMemory()
memory.human_prefix = "User"
memory.ai_prefix = "Bot"

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def parse_history(chat_history):
    return_string = """""" #Storing a empty string that we'll return
    toggle = True #So we can alternate between user and AI
    for i in chat_history.messages[-5:]: #We take the last 3 messages for the context window
        if i.type == 'human': 
            user = 'User: '
        else:
            user = 'ai: '
        return_string = return_string + user + str(i.content) +'\n'
    #print(return_string)
    return return_string
    
    
def query(case_id, query):
    chat_history = RedisChatMessageHistory(
        url=env('REDIS_URL'),
        session_id='-history',
        key_prefix=case_id,
        ttl=3600*24)
    
    conversation = ConversationChain(
        llm=titan_llm, 
        verbose=True, 
        memory=memory
        )
    

    conversation.prompt.template = """
        System: The following is a friendly conversation between a knowledgeable helpful assistant and a customer. 
        Only answer as the bot, never as the customer. The assistant is talkative and provides lots of specific details from it's context.\n\n
        Current conversation:\n {history}"""+ str(parse_history(chat_history=chat_history))+"""\nUser: {input}\nBot:
        """

    try:
        result =  conversation.predict(input=query)
        chat_history.add_user_message(query)
        chat_history.add_ai_message(result)
        tokens_used = num_tokens_from_string(conversation.prompt.template)
        print("####### TOKEN COUNT #######")
        print('total tokens used: ', tokens_used)
        print('query tokens: ', num_tokens_from_string(query))
        print('Result tokens: ', num_tokens_from_string(result))
        print("Chat history: ",num_tokens_from_string(parse_history(chat_history=chat_history)))
        return result

    except ValueError as error:
        if  "AccessDeniedException" in str(error):
            print(f"\x1b[41m{error}\
            \nTo troubeshoot this issue please refer to the following resources.\
            \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
            \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")      
            class StopExecution(ValueError):
                def _render_traceback_(self):
                    pass
            raise StopExecution        
        else:
            raise error
    return "something went wrong"
    
    
test = query(case_id='yes', query='What should I do when visiting Amsterdam?')
print(test)