import json

def is_valid_json(json_string):
    try:
        json_object = json.loads(json_string)
    except json.JSONDecodeError:
        return False
    return True

def readFile(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        # 逐行读取
        lines = file.readlines()
        
    result = []
    curr_line = ""

    for line in lines:
        if is_valid_json(line.strip()):
            print(line.strip())
        elif line.strip().startswith("{"):
            curr_line = line.strip()
        elif line.strip().endswith("}") and curr_line.startswith("{"):
            curr_line += line.strip()
            if is_valid_json(curr_line.strip()):
                print(curr_line)
            curr_line = ""
        elif curr_line.strip().startswith("{"):
            curr_line += line.strip()

readFile("outabc.txt")
