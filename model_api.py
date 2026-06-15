ATTACK_MODEL = "deepseek-v4-flash"
OPENAI_BASE_URL = "https://api.shredder.money/v1"
OPENAI_API_KEY = "sk-X02T7i7OLSfSChqz49u8vc4xeLW4xHbzeTYyuBi1kV6mZwT0"
OPTIMIZER_MODEL = "deepseek-v4-flash"
try:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model = ATTACK_MODEL, 
        temperature = 0.5,
        base_url =OPENAI_BASE_URL,
        api_key = OPENAI_API_KEY,
        streaming = True
    )
except ImportError:
    llm = None
if __name__ == "__main__":
    if llm is None:
        raise RuntimeError("Install langchain_openai to use llm.invoke")
    print(llm.invoke("hello"))
        