import google.generativeai as genai
genai.configure(api_key="AIzaSyDt68rUvLeTJsWUSB607Zp40Ioi0uF-JxI")
for m in genai.list_models():
    print(m.name)
