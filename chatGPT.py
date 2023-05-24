# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 15:23:15 2023

@author: sebas
"""


    
    
import openai

openai.api_key = "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"


while True:
    
    prompt = input("\nIntroduce una pregunta: ")
    
    if prompt == "exit":
        break
    
    completion = openai.Completion.create(engine="text-davinci-003",
                             prompt = prompt,
                             max_tokens=2048)
    
    print(completion.choices[0].text)