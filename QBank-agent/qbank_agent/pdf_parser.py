import json
import pdfplumber
from pathlib import Path
from . import config

def extract_text_from_pdf(pdf_path: Path):
    """
    Extract text and titles from each page of a PDF document using pdfplumber.
    """
    slides_data = []
    
    print(f"   📄 开始解析 PDF 页面文本和标题提取...")
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"   📊 共发现 {total_pages} 页内容")
        for i, page in enumerate(pdf.pages):
            print(f"\r      ... 正在提取第 {i + 1}/{total_pages} 页", end="", flush=True)
            text = page.extract_text()
            if not text:
                continue
                
            largest_size = 0
            title_text = "Untitled Slide"
            
            # Simple line-based fallback for titles
            lines = text.split('\n')
            if lines:
                possible_title = lines[0].strip()
                if len(possible_title) < 100:
                    title_text = possible_title

            # More robust identification based on layout
            words = page.extract_words()
            if words:
                words.sort(key=lambda w: (round(w['top']), w['x0']))
                current_top = -1
                current_line_words = []
                lines_with_pos = []
                
                for w in words:
                    if current_top == -1:
                        current_top = w['top']
                        current_line_words.append(w)
                    elif abs(w['top'] - current_top) < 5:
                        current_line_words.append(w)
                    else:
                        lines_with_pos.append(current_line_words)
                        current_line_words = [w]
                        current_top = w['top']
                if current_line_words:
                    lines_with_pos.append(current_line_words)
                
                if lines_with_pos:
                    first_line = " ".join([w['text'] for w in lines_with_pos[0]])
                    if len(first_line) < 150:
                        title_text = first_line

            # Clean content from title repetition
            content = text
            if title_text in content:
                content = content.replace(title_text, "", 1).strip()

            slides_data.append({
                "file": pdf_path.name,
                "page": i + 1,
                "page_range": [i + 1],
                "title": title_text,
                "content": content
            })

    print("\n   ✅ 第1阶段: 页面文本与标题提取完成")
    return slides_data

def merge_slides(all_slides):
    """Merge pages with the same title from the same file."""
    print(f"   🔄 开始合并相同标题的幻灯片 (相邻页面)...")
    merged = []
    if not all_slides:
        print("   ⚠️ 没有可合并的幻灯片数据")
        return merged

    current_slide = all_slides[0]
    
    for next_slide in all_slides[1:]:
        if (next_slide['title'] == current_slide['title']) and (next_slide['file'] == current_slide['file']):
            current_slide['content'] += "\n\n" + next_slide['content']
            current_slide['page_range'].extend(next_slide['page_range'])
        else:
            merged.append(current_slide)
            current_slide = next_slide
    
    merged.append(current_slide)
    print(f"   ✅ 第2阶段: 页面合并完成，将 {len(all_slides)} 页压缩为 {len(merged)} 个独立知识块")
    return merged

def process_and_save_pdf(pdf_path: Path, output_dir: Path):
    """Processes a single PDF slide deck and outputs a parsed JSON mapping."""
    output_path = output_dir / f"{pdf_path.stem}.json"
    
    if output_path.exists():
        print(f"   ⏭️  解析结果已存在，跳过解析: {output_path.name}")
        return output_path

    print("\n" + "="*60)
    print(f"🚀 [任务开始] 处理文件: {pdf_path.name}")
    print("="*60)
    raw_slides = extract_text_from_pdf(pdf_path)
    merged_slides = merge_slides(raw_slides)
    
    outline = [slide['title'] for slide in merged_slides]
    outline_with_page_range = [{"title": slide['title'], "page_range": slide['page_range']} for slide in merged_slides]
    
    for slide in merged_slides:
        if 'page_range' in slide:
            del slide['page_range']

    final_output = {
        "source_file": pdf_path.name,
        "outline": outline,
        "outline_with_page_range": outline_with_page_range,
        "slides": merged_slides
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    
    print(f"   💾 解析结果已成功保存至: {output_path}")
    print(f"🎯 [任务完成] {pdf_path.name} 处理完毕")
    
    return output_path

def process_all_slides(input_dir: Path = config.SLIDES_DIR, output_dir: Path = config.JSONS_DIR):
    """Runs the parser pipeline for all PDFs in the input directory."""
    config.setup_directories()
    pdf_files = sorted(list(input_dir.glob("*.pdf")))
    results = []

    total_files = len(pdf_files)
    print(f"\n📂 开始批量处理 PDF 文件...")
    print(f"   输入目录: {input_dir}")
    print(f"   输出目录: {output_dir}")
    print(f"   共发现 {total_files} 个 PDF 文件待处理")

    for idx, pdf_file in enumerate(pdf_files):
        print(f"\n⏳ 进度: {idx + 1} / {total_files}")
        try:
            output_file = process_and_save_pdf(pdf_file, output_dir)
            results.append(output_file)
        except Exception as e:
            print(f"❌ 处理失败 {pdf_file.name}: {e}")

    print("\n" + "="*60)
    print(f"🎉 批量处理完毕！共成功处理 {len(results)}/{total_files} 个文件。")
    print("="*60 + "\n")
    return results
