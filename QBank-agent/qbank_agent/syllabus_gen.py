import json
from pathlib import Path
from openai import OpenAI
from . import config

# --- PROMPT TEMPLATES ---
SYSTEM_PROMPT = """You are an expert professor and curriculum designer in the field of Electrical Engineering and Pattern Recognition. 
Your task is to analyze a list of slide titles and their corresponding page numbers from a lecture slide deck to construct a high-level, well-structured syllabus."""

USER_PROMPT_TEMPLATE = """
Here is the outline of the lecture slides "{source_file}":
{outline_json}

Please generate a high-level syllabus in JSON format based on these slides. 
Follow these rules:
1. **Group Related Slides**: Identify the major topics covered. Group consecutive or related slides into a single high-level "Lecture Topic".
2. **Synthesize Sub-points**: For each major topic, summarize the key knowledge points (concepts, algorithms, theories) covered. Do not just list slide titles; synthesize them into 3-5 key sub-points.
3. **Determine Page Ranges**: For each major topic, calculate the start and end page numbers based on the input slides that belong to that topic.
4. **JSON Output**: The output must be valid JSON with the following structure:
{{
  "syllabus": [
    {{
      "topic": "Major Topic Name",
      "page_range": [start_page_int, end_page_int],
      "sub_points": [
        "Key Concept 1",
        "Key Concept 2",
        "..."
      ]
    }},
    ...
  ]
}}
"""

def generate_syllabus_for_file(client, json_file: Path, output_dir: Path):
    """Reads a parsed JSON from pdf_parser and creates a syllabus using OpenAI."""
    output_path = output_dir / f"{json_file.stem}_syllabus.json"
    
    if output_path.exists():
        print(f"   ⏭️  大纲文件已存在，跳过生成: {output_path.name}")
        return output_path

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        source_file = data.get('source_file', json_file.name)
        outline = data.get('outline_with_page_range', [])

        if not outline:
            print(f"   ⚠️  警告：未在 {json_file.name} 中找到大纲信息，跳过。")
            return None

        user_message = USER_PROMPT_TEMPLATE.format(
            source_file=source_file,
            outline_json=json.dumps(outline, indent=2)
        )

        print(f"   🤖 请求大模型生成课程大纲... (依赖文件: {json_file.name})", end="", flush=True)
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"}
        )
        print(" 完成!")

        content = response.choices[0].message.content
        syllabus_data = json.loads(content)
        syllabus_data["source_file"] = source_file
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(syllabus_data, f, indent=2, ensure_ascii=False)
            
        print(f"   💾 大纲已成功保存至: {output_path}")
        return output_path

    except Exception as e:
        print(f"   ❌ 处理失败 {json_file.name}: {e}")
        return None

def process_all_syllabus():
    """Generates a syllabus for all parsed slide JSONs."""
    config.setup_directories()
    
    if not config.OPENAI_API_KEY:
        print("❌ 错误：配置文件中未找到 OPENAI_API_KEY。")
        return []

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    json_files = sorted(list(config.JSONS_DIR.glob("*.json")))
    total_files = len(json_files)
    
    print(f"\n📖 开始批量生成课程大纲...")
    print(f"   AI 模型: {config.OPENAI_MODEL}")
    print(f"   共发现 {total_files} 个已解析的 PDF 结构文件待处理\n")
    
    results = []
    for idx, json_file in enumerate(json_files):
        print("="*60)
        print(f"🚀 [大纲任务] 处理文件: {json_file.name} ({idx + 1}/{total_files})")
        print("="*60)
        
        output_path = generate_syllabus_for_file(client, json_file, config.SYLLABUS_DIR)
        if output_path:
            results.append(output_path)
            
        print(f"🎯 [任务完成] {json_file.name} 大纲处理完毕\n")
            
    print("="*60)
    print(f"🎉 批量大纲生成完毕！共有效生成/跳过 {len(results)}/{total_files} 个大纲文件。")
    print("="*60 + "\n")
    return results
