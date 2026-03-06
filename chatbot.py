from langchain.messages import SystemMessage, HumanMessage, AIMessage
from rag import final_chain
import dotenv

dotenv.load_dotenv('.env')

chat_history = [
    SystemMessage(content="You are a helpful assistant.")
]

while True:
 user_input = input('You:')
 chat_history.append(HumanMessage(content=user_input))
 if user_input== 'exit':
  break
 result = final_chain.invoke(user_input)
 chat_history.append(AIMessage(content=result))
 print('AI:', result)