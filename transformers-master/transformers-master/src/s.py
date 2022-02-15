import json
s='阿萨的说法\n阿迪斯发的\n洒洒水'
with open('test.json','w',encoding='utf-8') as f:
    json.dump(s,f,ensure_ascii=False)