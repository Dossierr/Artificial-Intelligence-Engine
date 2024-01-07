import tiktoken

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def parse_history(chat_history):
    return_string = """""" #Storing a empty string that we'll return
    toggle = True #So we can alternate between user and AI
    for i in chat_history.messages[-5:]: #We take the last few messages for the context window
        if i.type == 'human': 
            user = 'User: '
        else:
            user = 'ai: '
        return_string = return_string + user + str(i.content) +'\n'
    #print(return_string)
    return return_string

def parse_relevant_documents(documents_list):
    return_string = """""" #Storing a empty string that we'll return
    sources = []
    j = 1 
    for i in documents_list: #We take the last few messages for the context window
        j = j + 1
        print("#### DOCUMENT: ", j, ' ####')
        print(i)
    return_string = return_string + str(i[0]) +'\n'
    #print(return_string)
    return return_string, sources