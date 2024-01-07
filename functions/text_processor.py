import boto3
import redis
import environ
from langchain.cache import RedisCache
from langchain.memory import RedisChatMessageHistory
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.globals import set_llm_cache
from langchain.llms.bedrock import Bedrock
from utils import num_tokens_from_string, parse_history, parse_relevant_documents
from vectorstore import chroma_index


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

#Log into bedrock as a client
client_boto3 = boto3.client(service_name="bedrock-runtime", 
                       aws_access_key_id=env('BEDROCK_AWS_ACCESS_KEY_ID'), 
                       aws_secret_access_key=env('BEDROCK_AWS_SECRET_KEY'))

#Defining the Model we want to use
modelId = "amazon.titan-text-express-v1"         
titan_llm = Bedrock(model_id=modelId, client=client_boto3, streaming=False)
titan_llm.model_kwargs = {'temperature': 0.1, }

#Setting memory for conversation
memory = ConversationBufferMemory()
memory.human_prefix = "User"
memory.ai_prefix = "Bot"

    
    
def query(case_id, query):
    database = chroma_index(case_id=case_id, query=query)
    chat_history = RedisChatMessageHistory(
        url=env('REDIS_URL'),
        session_id='-history',
        key_prefix=case_id,
        ttl=3600*24)
    retrieved_documents = database.similarity_search_with_score(query)
    retrieved_text, retrieved_sources = parse_relevant_documents(retrieved_documents)
    print('#### RETRIEVED DOCUMENTS #### \n', retrieved_text)
    print("### LENGHT ###", len(retrieved_documents))
    print("### SOURCES ###", retrieved_sources)

    conversation = ConversationChain(
        llm=titan_llm, 
        verbose=True, 
        memory=memory,
        )
    

    conversation.prompt.template = """
        System: The following is a friendly conversation between a knowledgeable helpful assistant and a customer. 
        Only answer as the bot, never as the customer, don't prepend your answer with AI:. The assistant is talkative and provides lots of specific details from it's context.\n\n
        Current conversation:\n {history}"""+ str(parse_history(chat_history=chat_history))+"""\nUser: {input}\n
        Bot:
        """

    try:
        result =  conversation.predict(input=query)
        chat_history.add_user_message(query)
        chat_history.add_ai_message(result)
        prompt_tokens = num_tokens_from_string(conversation.prompt.template)
        query_tokens = num_tokens_from_string(query)
        result_tokens = num_tokens_from_string(result)
        retrieved_document_tokens = 200
        chat_history_tokens = num_tokens_from_string(parse_history(chat_history=chat_history))
        documents_retrieved = ['test.pdf', 'document.pdf']
        result = {
            'answer': result,
            'total_tokens': prompt_tokens+query_tokens+result_tokens+chat_history_tokens + retrieved_document_tokens,
            'prompt_tokens': prompt_tokens,
            'query_tokens': query_tokens,
            'result_tokens': result_tokens,
            'chat_history_tokens': chat_history_tokens,
            'documents_retrieved': documents_retrieved,
            'document_retrieved_tokens': retrieved_document_tokens
        }
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
    
    
test = query(case_id='testfolder', query='Wat zegt de VVD  over het milieu?')
print(test)