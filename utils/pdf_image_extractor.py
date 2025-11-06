"""
Simple PDF image extractor using LlamaIndex and PyMuPDF
No expensive GPT-4 Vision calls - just extracts and describes images
"""
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image
import base64
from io import BytesIO
import hashlib
import pytesseract  # NEW

from llama_index.core import Document
from llama_index.core.schema import ImageDocument
import fitz  # PyMuPDF

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import numpy as np

console = Console()


class PDFImageExtractor:
    """Extract images from PDFs and include them in document processing"""
    
    def __init__(self, extract_images: bool = True, min_image_size: int = 10000):
        """
        Args:
            extract_images: Whether to extract images
            min_image_size: Minimum image size in bytes to extract
        """
        self.extract_images = extract_images
        self.min_image_size = min_image_size
        self.temp_image_dir = Path("./temp_pdf_images")
        self.temp_image_dir.mkdir(parents=True, exist_ok=True)
        
        # NEW: Check OCR availability
        try:
            pytesseract.get_tesseract_version()
            self.ocr_available = True
        except:
            self.ocr_available = False
            console.print("[yellow]⚠ Tesseract OCR not found - OCR filtering disabled[/yellow]")

    def _has_meaningful_text(self, image_path: Path, min_text_length: int = 10) -> bool:
        """Use OCR to detect if image contains meaningful text - SAME AS MULTIMODAL"""
        if not self.ocr_available:
            return None
        
        try:
            with Image.open(image_path) as img:
                if img.mode not in ['RGB', 'L']:
                    img = img.convert('RGB')
                
                text = pytesseract.image_to_string(img, config='--psm 6')
                cleaned_text = ''.join(c for c in text if c.isalnum() or c.isspace())
                text_length = len(cleaned_text.strip())
                
                return text_length >= min_text_length
                
        except Exception as e:
            return None

    def _is_meaningful_image(self, image_path: Path) -> bool:
        """Filter out logos and small decorative images - ENHANCED with OCR"""
        try:
            file_size = image_path.stat().st_size
            if file_size < self.min_image_size:
                return False
            
            with Image.open(image_path) as img:
                if img.mode not in ['RGB', 'RGBA']:
                    img = img.convert('RGB')
                
                width, height = img.size
                
                # Filter 1: Minimum dimensions
                if width < 100 or height < 100:
                    return False
                
                # Filter 2: Aspect ratio
                aspect_ratio = max(width, height) / min(width, height)
                if aspect_ratio > 5.0:
                    return False
                
                # NEW: Filter 3: OCR text detection (PRIORITY FILTER)
                has_text = self._has_meaningful_text(image_path, min_text_length=10)
                if has_text:
                    return True  # Accept immediately if contains text
                
                # Filter 4: White background check
                img_small = img.resize((50, 50), Image.Resampling.LANCZOS)
                img_array = np.array(img_small)
                
                if img.mode == 'RGBA':
                    rgb_array = img_array[:, :, :3]
                else:
                    rgb_array = img_array
                
                light_pixels = np.all(rgb_array > 200, axis=2)
                light_percentage = np.sum(light_pixels) / light_pixels.size
                
                if light_percentage > 0.30:
                    return True
                
                # Filter 5: Color variance
                std_dev = np.std(rgb_array)
                if std_dev < 30:
                    return False
                
                # Filter 6: Unique colors
                flattened = rgb_array.reshape(-1, 3)
                unique_colors = len(np.unique(flattened, axis=0))
                if unique_colors < 50:
                    return False
                
                return True
                
        except Exception as e:
            console.print(f"[yellow]⚠ Error analyzing image {image_path}: {e}[/yellow]")
            return False

    def extract_images_from_pdf(self, pdf_path: Path) -> List[Dict]:
        """Extract meaningful images from PDF (filters out logos)"""
        if not self.extract_images:
            return []
        
        images_info = []
        
        try:
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Generate unique filename
                        image_hash = hashlib.md5(image_bytes).hexdigest()[:8]
                        image_filename = f"{pdf_path.stem}_p{page_num+1}_img{img_index+1}_{image_hash}.{image_ext}"
                        image_path = self.temp_image_dir / image_filename
                        
                        # Save image temporarily
                        with open(image_path, 'wb') as img_file:
                            img_file.write(image_bytes)
                        
                        # FILTER: Check if meaningful
                        if not self._is_meaningful_image(image_path):
                            image_path.unlink()  # Delete filtered image
                            continue
                        
                        # Get dimensions
                        try:
                            with Image.open(image_path) as pil_img:
                                width, height = pil_img.size
                        except:
                            width, height = 0, 0
                        
                        images_info.append({
                            'pdf_path': str(pdf_path),
                            'pdf_name': pdf_path.name,
                            'page_number': page_num + 1,
                            'image_index': img_index + 1,
                            'image_path': image_path,
                            'image_size': len(image_bytes),
                            'width': width,
                            'height': height,
                            'format': image_ext
                        })
                        
                    except Exception as e:
                        console.print(f"[yellow]⚠ Error extracting image {img_index} from page {page_num}: {e}[/yellow]")
                        continue
            
            pdf_document.close()
            
            if images_info:
                console.print(f"[green]✓ Extracted {len(images_info)} meaningful images from {pdf_path.name}[/green]")
            
        except Exception as e:
            console.print(f"[red]✗ Error processing PDF {pdf_path}: {e}[/red]")
        
        return images_info
    
    def create_image_documents(self, images_info: List[Dict]) -> List[Document]:
        """Create documents from extracted images with metadata
        
        Args:
            images_info: List of image info dictionaries
            
        Returns:
            List of Document objects with image information
        """
        documents = []
        
        for img_info in images_info:
            # Create a text description for the image
            text_content = f"""
[IMAGE FOUND IN DOCUMENT]
Source: {img_info['pdf_name']}
Page: {img_info['page_number']}
Image #{img_info['image_index']}
Dimensions: {img_info['width']}x{img_info['height']} pixels
Size: {img_info['image_size']:,} bytes
Format: {img_info['format'].upper()}

Note: This is an image extracted from the PDF. The actual visual content cannot be processed without OCR or vision models.
"""
            
            # Create document with rich metadata
            doc = Document(
                text=text_content.strip(),
                metadata={
                    'file_name': img_info['pdf_name'],
                    'page_label': str(img_info['page_number']),
                    'page_number': img_info['page_number'],
                    'content_type': 'image',
                    'processing_method': 'image_extracted',
                    'image_path': str(img_info['image_path']),
                    'image_index': img_info['image_index'],
                    'image_width': img_info['width'],
                    'image_height': img_info['height'],
                    'image_size': img_info['image_size'],
                    'image_format': img_info['format']
                }
            )
            documents.append(doc)
        
        return documents
    
    def process_pdf_with_images(self, pdf_path: Path) -> tuple[List[Document], List[Document]]:
        """Process PDF: extract text AND images
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (text_documents, image_documents)
        """
        # Extract images
        images_info = self.extract_images_from_pdf(pdf_path)
        
        # Create image documents
        image_documents = []
        if images_info:
            image_documents = self.create_image_documents(images_info)
        
        # Note: Text extraction is handled by SimpleDirectoryReader in docs_processor.py
        # We just return image documents here
        
        return [], image_documents  # Text docs handled separately
    
    def batch_process_pdfs(self, pdf_paths: List[Path]) -> List[Document]:
        """Process multiple PDFs and extract all images
        
        Args:
            pdf_paths: List of PDF file paths
            
        Returns:
            List of all image documents
        """
        all_image_docs = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                f"Extracting images from {len(pdf_paths)} PDFs...",
                total=len(pdf_paths)
            )
            
            for pdf_path in pdf_paths:
                _, image_docs = self.process_pdf_with_images(pdf_path)
                all_image_docs.extend(image_docs)
                progress.update(task, advance=1)
        
        console.print(f"\n[bold green]✅ Total images extracted: {len(all_image_docs)}[/bold green]")
        
        return all_image_docs
    
    def cleanup(self):
        """Clean up temporary image files"""
        try:
            if self.temp_image_dir.exists():
                for file in self.temp_image_dir.glob("*"):
                    if file.is_file():
                        file.unlink()
                console.print("[dim]Cleaned up temporary image files[/dim]")
        except Exception as e:
            console.print(f"[yellow]⚠ Error cleaning up temp files: {e}[/yellow]")