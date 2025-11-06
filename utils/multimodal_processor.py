"""
Multi-modal processor using LlamaIndex's GPT-4 Vision for images, tables, and charts
ENHANCED: Advanced detection for flowcharts, tables, and process diagrams
"""
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import base64
from io import BytesIO
import fitz  # PyMuPDF
import numpy as np
import pytesseract
import cv2

from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.schema import ImageDocument, ImageNode
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from config import Config
from logger_config import setup_logger

console = Console()


class MultiModalProcessor:
    """Process images, tables, and charts using GPT-4 Vision via LlamaIndex"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger('MultiModalProcessor', config.log_dir)
        
        # Use correct vision-capable model
        vision_model = getattr(config, 'multimodal_model', 'gpt-4o')
        if not any(v in vision_model for v in ['gpt-4o', 'gpt-4-vision']):
            vision_model = 'gpt-4o'
        
        self.vision_llm = OpenAIMultiModal(
            model=vision_model,
            api_key=config.openai_api_key,
            max_new_tokens=getattr(config, 'multimodal_max_tokens', 1024),
        )
        
        self.temp_image_dir = Path("./temp_multimodal_images")
        self.temp_image_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure Tesseract
        try:
            pytesseract.get_tesseract_version()
            self.ocr_available = True
            self.logger.info("Tesseract OCR available for document detection")
            console.print("[green]✓ OCR enabled - Advanced document detection[/green]")
        except Exception as e:
            self.ocr_available = False
            self.logger.warning(f"Tesseract OCR not available: {e}")
            console.print("[yellow]⚠ OCR not found - Install for better filtering[/yellow]")
        
        self.logger.info(f"Multi-modal processor initialized with {vision_model}")
        console.print(f"[green]✓ Multi-modal processor ready - ADVANCED filtering[/green]")

    def _detect_document_structure(self, image_path: Path) -> Tuple[bool, str]:
        """ADVANCED: Detect if image is a structured document (flowchart/table/diagram)
        
        Analyzes:
        1. Text density and distribution
        2. Box/rectangle structures
        3. Line patterns (grid, flow arrows)
        4. Text-to-graphics ratio
        5. Checkmarks/symbols
        
        Returns:
            (is_document, document_type) - True if structured document, type description
        """
        try:
            # Read image with OpenCV for advanced processing
            img_cv = cv2.imread(str(image_path))
            if img_cv is None:
                return False, "unreadable"
            
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # === CRITERION 1: Text Detection with OCR ===
            has_meaningful_text = False
            text_regions = 0
            
            if self.ocr_available:
                try:
                    # Get bounding boxes of text
                    d = pytesseract.image_to_data(Image.open(image_path), output_type=pytesseract.Output.DICT)
                    
                    # Count text regions with confidence > 60
                    text_regions = sum(1 for conf in d['conf'] if int(conf) > 60)
                    
                    # Get total text
                    all_text = ' '.join([d['text'][i] for i in range(len(d['text'])) if int(d['conf'][i]) > 60])
                    words = [w for w in all_text.split() if len(w) >= 2]
                    
                    has_meaningful_text = len(words) >= 5  # At least 5 words
                    
                    if has_meaningful_text:
                        self.logger.debug(f"✓ OCR: {len(words)} words, {text_regions} text regions in {image_path.name}")
                    
                except Exception as e:
                    self.logger.debug(f"OCR failed: {e}")
            
            # === CRITERION 2: Rectangle/Box Detection (flowchart boxes, table cells) ===
            # Edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Find contours (boxes)
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Count rectangular contours
            rectangles = 0
            for contour in contours:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                
                # Check if it's a rectangle (4 corners)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    
                    # Filter by size (must be meaningful, not tiny dots)
                    if w > 30 and h > 20:
                        rectangles += 1
            
            has_boxes = rectangles >= 3  # Flowcharts/tables have multiple boxes
            
            if has_boxes:
                self.logger.debug(f"✓ Structure: {rectangles} rectangular boxes in {image_path.name}")
            
            # === CRITERION 3: Line Detection (grid lines, flow arrows) ===
            # Hough Line Transform for straight lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
            
            horizontal_lines = 0
            vertical_lines = 0
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Check if horizontal (y1 ≈ y2)
                    if abs(y2 - y1) < 10:
                        horizontal_lines += 1
                    
                    # Check if vertical (x1 ≈ x2)
                    if abs(x2 - x1) < 10:
                        vertical_lines += 1
            
            has_grid = horizontal_lines >= 2 and vertical_lines >= 2
            
            if has_grid:
                self.logger.debug(f"✓ Grid: {horizontal_lines}H, {vertical_lines}V lines in {image_path.name}")
            
            # === CRITERION 4: White Background Check ===
            # Sample pixels from corners and edges (typical document backgrounds)
            border_pixels = np.concatenate([
                gray[0, :],      # Top edge
                gray[-1, :],     # Bottom edge
                gray[:, 0],      # Left edge
                gray[:, -1]      # Right edge
            ])
            
            white_bg_percentage = np.sum(border_pixels > 220) / len(border_pixels)
            has_white_bg = white_bg_percentage > 0.70  # 70% of border is white
            
            # === CRITERION 5: Red Object Detection (reject safety equipment) ===
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            
            # Red color range (in HSV)
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = mask1 + mask2
            
            red_percentage = np.sum(red_mask > 0) / (width * height)
            is_red_object = red_percentage > 0.20  # More than 20% red pixels
            
            if is_red_object:
                self.logger.debug(f"✗ Red object detected: {red_percentage:.1%} red pixels in {image_path.name}")
            
            # === DECISION LOGIC ===
            
            # REJECT: Red safety equipment (even if has structure)
            if is_red_object:
                self.logger.info(f"✗ REJECTED (red safety equipment): {image_path.name}")
                return False, "red_equipment"
            
            # ACCEPT: Flowchart (boxes + text + white background)
            if has_boxes and has_meaningful_text and has_white_bg:
                self.logger.info(f"✓✓✓ ACCEPTED (flowchart/diagram): {image_path.name}")
                return True, "flowchart"
            
            # ACCEPT: Table (grid + text + white background)
            if has_grid and has_meaningful_text and has_white_bg:
                self.logger.info(f"✓✓✓ ACCEPTED (table/grid): {image_path.name}")
                return True, "table"
            
            # ACCEPT: Text document (high text density + white background)
            if text_regions >= 10 and has_white_bg:
                self.logger.info(f"✓✓ ACCEPTED (text document): {image_path.name}")
                return True, "document"
            
            # REJECT: No document structure found
            self.logger.debug(f"✗ No document structure: text={has_meaningful_text}, boxes={has_boxes}, grid={has_grid}, white={has_white_bg}")
            return False, "no_structure"
            
        except Exception as e:
            self.logger.error(f"Error detecting document structure: {e}")
            return False, "error"

    def _is_meaningful_image(self, image_path: Path, min_size_bytes: int = 15000) -> bool:
        """ENHANCED filtering using advanced document structure detection
        
        Strategy:
        1. Basic size/dimension filters
        2. ADVANCED: Detect structured documents (flowcharts, tables, diagrams)
        3. REJECT: Red safety equipment and non-documents
        
        Args:
            image_path: Path to image file
            min_size_bytes: Minimum file size (increased to 15KB)
            
        Returns:
            True if structured document, False otherwise
        """
        try:
            # Filter 1: File size
            file_size = image_path.stat().st_size
            if file_size < min_size_bytes:
                self.logger.debug(f"✗ Too small: {image_path.name} ({file_size} bytes)")
                return False
            
            # Filter 2: Minimum dimensions
            with Image.open(image_path) as img:
                width, height = img.size
                
                MIN_WIDTH = 200  # Increased for documents
                MIN_HEIGHT = 150
                
                if width < MIN_WIDTH or height < MIN_HEIGHT:
                    self.logger.debug(f"✗ Too small dimensions: {image_path.name} ({width}x{height})")
                    return False
                
                # Aspect ratio check
                aspect_ratio = max(width, height) / min(width, height)
                if aspect_ratio > 4.0:
                    self.logger.debug(f"✗ Bad aspect ratio: {image_path.name} ({aspect_ratio:.2f})")
                    return False
            
            # Filter 3: ADVANCED document structure detection
            is_document, doc_type = self._detect_document_structure(image_path)
            
            if is_document:
                self.logger.info(f"✓✓✓ ACCEPTED ({doc_type}): {image_path.name}")
                return True
            else:
                self.logger.info(f"✗ REJECTED ({doc_type}): {image_path.name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error analyzing image {image_path}: {e}")
            return False

    def extract_images_from_pdf(self, pdf_path: Path) -> List[Dict]:
        """Extract ONLY structured documents (flowcharts, tables, diagrams)"""
        images_info = []
        rejected_count = 0
        rejection_reasons = {
            'too_small': 0,
            'red_equipment': 0,
            'no_structure': 0,
            'error': 0
        }
        
        try:
            pdf_document = fitz.open(pdf_path)
            total_extracted = 0
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                image_list = page.get_images(full=True)
                total_extracted += len(image_list)
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        image_filename = f"{pdf_path.stem}_p{page_num+1}_img{img_index+1}.{image_ext}"
                        image_path = self.temp_image_dir / image_filename
                        
                        # Save image temporarily
                        with open(image_path, 'wb') as img_file:
                            img_file.write(image_bytes)
                        
                        # ADVANCED FILTERING
                        if not self._is_meaningful_image(image_path):
                            image_path.unlink()
                            rejected_count += 1
                            continue
                        
                        images_info.append({
                            'pdf_path': str(pdf_path),
                            'page_number': page_num + 1,
                            'image_index': img_index + 1,
                            'image_path': image_path,
                            'image_size': len(image_bytes),
                            'file_name': pdf_path.name
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Error extracting image {img_index} from page {page_num}: {e}")
                        rejection_reasons['error'] += 1
                        continue
            
            pdf_document.close()
            
            acceptance_rate = (len(images_info) / total_extracted * 100) if total_extracted > 0 else 0
            
            self.logger.info(f"Extracted {len(images_info)}/{total_extracted} documents from {pdf_path.name} ({acceptance_rate:.1f}% acceptance)")
            
            console.print(f"[cyan]📄 {pdf_path.name}:[/cyan]")
            console.print(f"   📊 Total images found: {total_extracted}")
            console.print(f"   ✓ Accepted (flowcharts/tables/diagrams): [green]{len(images_info)}[/green]")
            console.print(f"   ✗ Rejected: [red]{rejected_count}[/red]")
            console.print(f"   📈 Acceptance rate: [blue]{acceptance_rate:.0f}%[/blue]")
            
        except Exception as e:
            self.logger.error(f"Error processing PDF {pdf_path}: {e}")
        
        return images_info

    def _image_to_base64(self, image_path: Path) -> str:
        """Convert image file to base64 string"""
        try:
            with Image.open(image_path) as img:
                if img.mode not in ['RGB', 'RGBA']:
                    img = img.convert('RGB')
                
                max_size = 2000
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                buffered = BytesIO()
                img.save(buffered, format="JPEG", quality=95)
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                return f"data:image/jpeg;base64,{img_base64}"
                
        except Exception as e:
            self.logger.error(f"Error converting image to base64: {e}")
            raise

    def process_image_with_vision(
        self, 
        image_path: Path,
        page_number: int = None,
        processing_type: str = 'general'
    ) -> Dict:
        """Process a single image using GPT-4o Vision"""
        try:
            if processing_type == 'table':
                prompt = self._get_table_prompt()
            elif processing_type == 'chart':
                prompt = self._get_chart_prompt()
            elif processing_type == 'ocr':
                prompt = self._get_ocr_prompt()
            else:
                prompt = self._get_general_prompt()
            
            image_data_url = self._image_to_base64(image_path)
            image_node = ImageNode(image_url=image_data_url)
            
            response = self.vision_llm.complete(
                prompt=prompt,
                image_documents=[image_node]
            )
            
            result = {
                'image_path': str(image_path),
                'page_number': page_number,
                'extracted_content': response.text,
                'processing_type': processing_type,
                'processing_method': 'gpt4_vision',
                'success': True,
                'word_count': len(response.text.split()),
                'confidence': 95
            }
            
            self.logger.info(f"Processed {image_path.name} ({processing_type})")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            return {
                'image_path': str(image_path),
                'page_number': page_number,
                'error': str(e),
                'processing_type': processing_type,
                'success': False
            }

    def _get_general_prompt(self) -> str:
        return """Analyze this image comprehensively. Extract and describe:

1. **Text Content**: All readable text, preserving formatting and structure
2. **Tables**: Convert any tables to markdown format with proper alignment
3. **Charts/Graphs**: Describe type, data points, trends, and insights
4. **Diagrams**: Explain components, relationships, and flow
5. **Visual Elements**: Any important visual information

Provide a clear, structured response that captures all information from the image."""

    def _get_table_prompt(self) -> str:
        return """This image contains a table. Please:

1. Convert the table to clean markdown format
2. Preserve all data accurately, including column headers, row labels, and all cell values
3. Maintain proper alignment
4. Include any table title or caption
5. Note any footnotes or annotations

Output the markdown table followed by any relevant notes."""

    def _get_chart_prompt(self) -> str:
        return """This image contains a chart or graph. Please provide:

1. **Chart Type**: Identify the type (bar, line, pie, scatter, etc.)
2. **Title and Labels**: Extract chart title, axis labels, legend
3. **Data Points**: List all data points and values visible
4. **Trends**: Describe key trends, patterns, and outliers
5. **Insights**: Provide main insights and conclusions
6. **Data Table**: Convert the chart data to a markdown table if applicable

Provide a comprehensive analysis that captures all information from the chart."""

    def _get_ocr_prompt(self) -> str:
        return """Extract all text from this image:

1. Preserve the original text structure and formatting
2. Maintain paragraph breaks and line spacing
3. Keep any bullet points, numbering, or lists
4. Extract text from any embedded elements
5. Note any text that is unclear or partially visible

Provide the extracted text in a clean, readable format."""

    def classify_image_content(self, image_path: Path) -> str:
        """Classify what type of content is in the image"""
        try:
            classify_prompt = """Classify this image into ONE category:
- 'table' if it contains primarily tabular data
- 'chart' if it contains graphs, charts, or plots
- 'text' if it contains primarily text (paragraphs, documents)
- 'diagram' if it contains flowcharts, diagrams, or schematics
- 'mixed' if it contains multiple types

Respond with only the category name."""
            
            image_data_url = self._image_to_base64(image_path)
            image_node = ImageNode(image_url=image_data_url)
            
            response = self.vision_llm.complete(
                prompt=classify_prompt,
                image_documents=[image_node]
            )
            
            classification = response.text.strip().lower()
            
            valid_types = ['table', 'chart', 'text', 'diagram', 'mixed']
            for vtype in valid_types:
                if vtype in classification:
                    return vtype
            
            return 'mixed'
            
        except Exception as e:
            self.logger.error(f"Error classifying image {image_path}: {e}")
            return 'mixed'

    def process_pdf_with_multimodal(self, pdf_path: Path) -> List[Document]:
        """Process entire PDF with multi-modal analysis"""
        console.print(f"\n[bold cyan]🎨 Multi-modal processing: {pdf_path.name}[/bold cyan]")
        
        images_info = self.extract_images_from_pdf(pdf_path)
        
        if not images_info:
            console.print(f"[yellow]No charts/tables/diagrams found in {pdf_path.name}[/yellow]")
            return []
        
        console.print(f"[green]✓ Processing {len(images_info)} charts/tables/diagrams with GPT-4o Vision[/green]")
        
        documents = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing with GPT-4o Vision...", total=len(images_info))
            
            for img_info in images_info:
                image_path = img_info['image_path']
                page_number = img_info['page_number']
                
                content_type = self.classify_image_content(image_path)
                
                result = self.process_image_with_vision(
                    image_path=image_path,
                    page_number=page_number,
                    processing_type=content_type
                )
                
                if result.get('success'):
                    doc = Document(
                        text=result['extracted_content'],
                        metadata={
                            'file_name': img_info['file_name'],
                            'page_label': str(page_number),
                            'page_number': page_number,
                            'source_image': str(image_path),
                            'processing_method': 'multimodal_gpt4_vision',
                            'content_type': content_type,
                            'image_index': img_info['image_index'],
                            'word_count': result['word_count'],
                            'confidence': result['confidence']
                        }
                    )
                    documents.append(doc)
                    
                    progress.update(
                        task, 
                        advance=1,
                        description=f"Processed: {content_type} on page {page_number}"
                    )
                else:
                    progress.update(task, advance=1)
        
        console.print(f"[green]✅ Successfully processed {len(documents)} charts/tables/diagrams[/green]")
        
        return documents

    def batch_process_pdfs(self, pdf_paths: List[Path]) -> List[Document]:
        """Process multiple PDFs with multi-modal analysis"""
        all_documents = []
        
        console.print(f"\n[bold blue]🎨 Multi-Modal Batch Processing: {len(pdf_paths)} PDFs[/bold blue]")
        console.print(f"[dim]Using SMART filtering: Accepts charts/tables/diagrams, rejects logos[/dim]\n")
        
        for pdf_path in pdf_paths:
            docs = self.process_pdf_with_multimodal(pdf_path)
            all_documents.extend(docs)
        
        console.print(f"\n[bold green]✅ Total documents from charts/tables/diagrams: {len(all_documents)}[/bold green]")
        
        return all_documents

    def cleanup(self):
        """Clean up temporary image files"""
        try:
            if self.temp_image_dir.exists():
                for file in self.temp_image_dir.glob("*"):
                    if file.is_file():
                        file.unlink()
                self.logger.info("Cleaned up temporary multi-modal images")
        except Exception as e:
            self.logger.error(f"Error cleaning up temp files: {e}")