import json
import os
import glob

def update_questions_file():
    images_dir = r"images"
    questions_file = r"CVLM_questions.jsonl"
    
    # 获取所有图片文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp']
    image_files = []
    
    for ext in image_extensions:
        pattern = os.path.join(images_dir, ext)
        image_files.extend(glob.glob(pattern))
    
    image_filenames = [os.path.basename(f) for f in image_files]
    image_filenames = [f for f in image_filenames if not f.endswith('.txt')]
    
    print(f"找到 {len(image_filenames)} 个图片文件")
    
    # 提示词
    uniform_text = "I need to judge the authenticity of a news story based on this image, and your task is to describe the main content of the image in more detail in one sentence."
    
    # 生成新的问题数据
    new_questions = []
    for i, image_file in enumerate(image_filenames):
        relative_image_path = f"tw_dataset/twitter_dataset/devset/images/{image_file}"
        question_data = {
            "question_id": i + 100000,
            "image": relative_image_path,   # 使用相对路径
            "text": uniform_text,
            "category": "default"
        }
        new_questions.append(question_data)
    
    # 写入新的 JSONL 文件
    with open(questions_file, 'w', encoding='utf-8') as f:
        for question in new_questions:
            f.write(json.dumps(question) + '\n')
    
    print(f"已成功更新 {questions_file}")
    print(f"共生成 {len(new_questions)} 个问题")
    print(f"图片路径: {images_dir}")
    print(f"统一文本: {uniform_text}")
    
    print("\n前5个示例:")
    for i, question in enumerate(new_questions[:5]):
        print(f"{i+1}. {json.dumps(question, indent=2)}")

if __name__ == "__main__":
    update_questions_file()
