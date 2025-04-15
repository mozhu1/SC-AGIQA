import os
import csv
import base64
import imghdr  # 用于检测图像格式
from openai import OpenAI

# 初始化OpenAI客户端
client = OpenAI(
    api_key=<YOUR_API>,
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

# 支持的图像格式
SUPPORTED_IMAGE_FORMATS = ['jpeg', 'png']

# 读取CSV文件
def read_csv(file_path):
    # 检查文件是否存在，若不存在则返回空列表
    if not os.path.exists(file_path):
        return []
    # 以只读模式打开CSV文件
    with open(file_path, mode='r', encoding='utf-8') as f:
        # 使用DictReader读取CSV文件，将每一行转换为字典
        reader = csv.DictReader(f)
        return list(reader)

# 检测图像真实格式
def get_image_format(image_path):
    try:
        # 使用imghdr检测图像格式
        image_format = imghdr.what(image_path)
        return image_format
    except Exception as e:
        print(f"检测图像格式失败: {e}")
        return None

# 编码图像为base64
def encode_image(image_path):
    # 以二进制只读模式打开图像文件
    with open(image_path, "rb") as image_file:
        # 读取图像文件内容并进行Base64编码，然后解码为字符串
        return base64.b64encode(image_file.read()).decode('utf-8')

# 生成图像描述
def generate_description(image_base64, prompt):
    try:
        # 调用OpenAI的聊天完成API
        completion = client.chat.completions.create(
            model="doubao-1-5-vision-pro-32k-250115",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional image description generator. Please generate a description of approximately 50 words. Including its detailed content, style(anime style, sci - fi style, realistic style, abstract style, baroque style and so on), shooting Angle as well as color atmosphere and quality.Put the content at the top, be a little more detailed."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please describe this picture in 50 words.Your answer should not have superfluous words such as \"The image\", but rather start directly with your description"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        # 返回生成的描述内容
        return completion.choices[0].message.content
    except Exception as e:
        # 若API调用出错，打印错误信息并返回"Error"
        print(f"API调用出错: {e}")
        return "Error"

# 追加写入CSV
def append_to_csv(file_path, row):
    # 检查文件是否存在
    file_exists = os.path.exists(file_path)
    # 以追加模式打开CSV文件
    with open(file_path, mode='a', encoding='utf-8', newline='') as f:
        # 创建CSV写入器
        writer = csv.writer(f)
        # 若文件不存在，写入表头
        if not file_exists:
            writer.writerow(['name', 'answer'])
        # 写入数据行
        writer.writerow([row['name'], row['answer']])

def main():
    # 输入CSV文件路径
    input_csv = 'E:\\AGIQA-1k-Database-main\\file\\AIGC_all_shunxu.csv'
    # 输出CSV文件路径
    output_csv = 'E:\\AGIQA-1k-Database-main\\answer.csv'

    # 读取输入CSV文件中的数据
    data = read_csv(input_csv)
    # 读取输出CSV文件中已处理的图片名称
    processed = {row['name'] for row in read_csv(output_csv)}

    for row in data:
        # 获取图片名称
        image_name = row['name']
        # 若图片名称已在已处理列表中，跳过该图片
        if image_name in processed:
            print(f"已跳过: {image_name}")
            continue
        # 构建图片的完整路径
        image_path = os.path.join(os.path.dirname(input_csv), image_name)

        if os.path.exists(image_path):
            print(f"处理中: {image_name}...")
            try:
                # 检查图像格式
                image_format = get_image_format(image_path)
                if image_format is None:
                    print(f"图像格式检测失败: {image_name}")
                    continue
                print(f"检测到图像格式: {image_format}")

                # 检查图像格式是否支持
                if image_format.lower() not in SUPPORTED_IMAGE_FORMATS:
                    print(f"不支持的图像格式: {image_name} (格式: {image_format})")
                    continue

                # 对图片进行Base64编码
                image_base64 = encode_image(image_path)
                # 生成图片描述
                answer = generate_description(image_base64, '12345')
                # 将结果追加到输出CSV文件中
                append_to_csv(output_csv, {'name': image_name, 'answer': answer})
            except Exception as e:
                # 若处理过程中出错，打印错误信息
                print(f"处理失败: {image_name} - {str(e)}")
        else:
            # 若图片文件不存在，打印缺失信息
            print(f"图像缺失: {image_name}")

    print("处理完成")

if __name__ == "__main__":
    main()