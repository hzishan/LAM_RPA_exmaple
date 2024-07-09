
function_list = [
    "ManagerScedule",
    "SchedulePreference",
    "PersonalScheduleView",
    "LeaveRequest",
]

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].strip()
    lines = list(filter(lambda x: x != '', lines))
    return lines

def main():
    from pathlib import Path
    import re
    get_info = read_file(str(Path.cwd() / 'documents/leave_info.txt'))
    _inputs = []
    print("請填寫以下資訊：")
    for i in range(len(get_info)):
        inp = input(f"{get_info[i]}:")
        if inp == "" and re.match(r'\w+\(可選\)', get_info[i]):
            print(f"沒有{get_info[i]}")
            _inputs.append(inp)
        else: 
            while inp == "":
                print(f"{get_info[i]}不可為空")
                inp = input(f"{get_info[i]}:")
            _inputs.append(inp)
    info_dict = dict(zip(get_info, _inputs))
    print(info_dict)
    # {'申請人': 'a',
    #   '申請部門': 'IT',
    #   '代理人(可選)': '',
    #   '請假類型': '病假',
    #   '請假事由': '出車禍',
    #   '開始日期(xxxx.xx.xx)': '2024.05.17',
    #   '結束日期(可選)': '',
    #   '備註(可選)': ''} 


if __name__ == '__main__':
    import os, sys
    sys.path.append(os.getcwd())
    main()