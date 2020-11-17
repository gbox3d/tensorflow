#%%
from google.protobuf import text_format
from object_detection.protos import string_int_label_map_pb2


#%%
with open('./res/labelmap.pbtxt',"rt") as fd :
    # fd = open('./res/labelmap.pbtxt',"rt")
    _text = fd.read()

print(_text)
# %%
# 파싱 
label_map = string_int_label_map_pb2.StringIntLabelMap()
try:
    text_format.Merge(_text, label_map)
except text_format.ParseError:
    label_map.ParseFromString(_text)

# print(label_map)
_dic = {}
for item in label_map.item:
    print(item)
    print(item.name)
    print(item.id)
    _dic[item.name] = item.id

print(_dic)
# %%
