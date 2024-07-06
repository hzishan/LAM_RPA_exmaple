def run_sequential_chain():
    from model_setting import get_llm
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.chains.llm import LLMChain
    from langchain.chains.sequential import SequentialChain

    llm = get_llm()

    prompt_one = ChatPromptTemplate.from_template(
        "請幫我寫一首簡單的{song}歌。"
    )
    chain_one = LLMChain(llm=llm, prompt=prompt_one, output_key="lyric") # 注意這裡的 output_key

    prompt_two = ChatPromptTemplate.from_template(
        "請幫我把下面這首歌翻譯成{language}：{lyric}"
    )
    chain_two = LLMChain(llm=llm, prompt=prompt_two,
                         output_key="translated_lyric")  

    prompt_three = ChatPromptTemplate.from_template(
        "請幫我把下面這首歌用一句話做摘要：{language}："
    )
    chain_three = LLMChain(llm=llm, prompt=prompt_three,
                           output_key="summary")  
    
    overall_chain = SequentialChain(
        chains=[chain_one, chain_two, chain_three],
        input_variables=["song", "language"],
        output_variables=["lyric", "translated_lyric", "summary"],
        verbose=True
    )

    result = overall_chain({"song": "搖滾", "language": "日文"})
    print(result)


if __name__ == "__main__":
    import os, sys
    sys.path.append(os.getcwd())
    run_sequential_chain()