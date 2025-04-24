#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Markdown to WordPress HTML Converter

This script converts Markdown files to WordPress-compatible HTML format,
with special handling for LaTeX equations, images, and other elements.

Usage:
    python md_to_wordpress.py input.md output.wpml
"""

import re
import sys
import os
import argparse
import html
from bs4 import BeautifulSoup, NavigableString

try:
    from markdown import Markdown
except ImportError:
    print("Error: 'markdown' package not found. Please install it using:")
    print("pip install markdown")
    sys.exit(1)

def process_images(content):
    """
    Convert Markdown image syntax to WordPress HTML image tags.
    
    Args:
        content (str): The content with Markdown images
        
    Returns:
        str: Content with images converted to WordPress HTML
    """
    # Match Markdown image syntax: ![alt text](image_url)
    pattern = r'!\[(.*?)\]\((.*?)\)'
    
    def replace_image(match):
        alt_text = match.group(1)
        image_url = match.group(2)
        return f'<img src="{image_url}" alt="{alt_text}" class="size-full" />'
    
    return re.sub(pattern, replace_image, content)

def process_headings(soup):
    """
    Ensure headings have proper WordPress formatting.
    
    Args:
        soup (BeautifulSoup): BeautifulSoup object
        
    Returns:
        BeautifulSoup: Processed soup object
    """
    # Process all heading tags (h1 through h6)
    for i in range(1, 7):
        for heading in soup.find_all(f'h{i}'):
            # Ensure heading has proper WordPress formatting
            heading_text = heading.get_text()
            new_tag = soup.new_tag(f'h{i}')
            new_tag.string = heading_text
            heading.replace_with(new_tag)
    
    return soup

def process_lists(soup):
    """
    Ensure lists have proper WordPress formatting.
    
    Args:
        soup (BeautifulSoup): BeautifulSoup object
        
    Returns:
        BeautifulSoup: Processed soup object
    """
    # Process unordered lists
    for ul in soup.find_all('ul'):
        new_ul = soup.new_tag('ul')
        for li in ul.find_all('li'):
            new_li = soup.new_tag('li')
            new_li.append(NavigableString(li.get_text()))
            new_ul.append(new_li)
        ul.replace_with(new_ul)
    
    # Process ordered lists
    for ol in soup.find_all('ol'):
        new_ol = soup.new_tag('ol')
        for li in ol.find_all('li'):
            new_li = soup.new_tag('li')
            new_li.append(NavigableString(li.get_text()))
            new_ol.append(new_li)
        ol.replace_with(new_ol)
    
    return soup

def process_blockquotes(soup):
    """
    Ensure blockquotes have proper WordPress formatting.
    
    Args:
        soup (BeautifulSoup): BeautifulSoup object
        
    Returns:
        BeautifulSoup: Processed soup object
    """
    for blockquote in soup.find_all('blockquote'):
        new_blockquote = soup.new_tag('blockquote')
        for p in blockquote.find_all('p'):
            new_p = soup.new_tag('p')
            new_p.append(NavigableString(p.get_text()))
            new_blockquote.append(new_p)
        blockquote.replace_with(new_blockquote)
    
    return soup

def process_tables(soup):
    """
    Ensure tables have proper WordPress formatting.
    
    Args:
        soup (BeautifulSoup): BeautifulSoup object
        
    Returns:
        BeautifulSoup: Processed soup object
    """
    for table in soup.find_all('table'):
        new_table = soup.new_tag('table')
        
        # Process table header
        if table.find('thead'):
            thead = soup.new_tag('thead')
            tr = soup.new_tag('tr')
            
            for th in table.thead.find_all('th'):
                new_th = soup.new_tag('th')
                new_th.append(NavigableString(th.get_text()))
                tr.append(new_th)
            
            thead.append(tr)
            new_table.append(thead)
        
        # Process table body
        if table.find('tbody'):
            tbody = soup.new_tag('tbody')
            
            for tr in table.tbody.find_all('tr'):
                new_tr = soup.new_tag('tr')
                
                for td in tr.find_all('td'):
                    new_td = soup.new_tag('td')
                    new_td.append(NavigableString(td.get_text()))
                    new_tr.append(new_td)
                
                tbody.append(new_tr)
            
            new_table.append(tbody)
        
        table.replace_with(new_table)
    
    return soup

def process_links(soup):
    """
    Ensure links have proper WordPress formatting with target="_blank".
    
    Args:
        soup (BeautifulSoup): BeautifulSoup object
        
    Returns:
        BeautifulSoup: Processed soup object
    """
    for a in soup.find_all('a'):
        href = a.get('href', '')
        if href.startswith('http'):
            a['target'] = '_blank'
            a['rel'] = 'noopener noreferrer'
    
    return soup

def preprocess_latex(content):
    """
    Preprocess LaTeX equations in Markdown content to protect them from Markdown parsing.
    
    Args:
        content (str): Markdown content with LaTeX equations
        
    Returns:
        tuple: (processed content, dictionary of LaTeX blocks)
    """
    # Dictionary to store LaTeX blocks
    latex_blocks = {}
    
    # Process display LaTeX ($$...$$)
    def replace_display_latex(match):
        content = match.group(1)
        placeholder = f"LATEX_DISPLAY_{len(latex_blocks)}"
        latex_blocks[placeholder] = f"[latex display=true]\n{content}\n[/latex]"
        return placeholder
    
    content = re.sub(r'\$\$(.*?)\$\$', replace_display_latex, content, flags=re.DOTALL)
    
    # Process inline LaTeX ($...$) - careful not to match $$
    def replace_inline_latex(match):
        content = match.group(1)
        placeholder = f"LATEX_INLINE_{len(latex_blocks)}"
        latex_blocks[placeholder] = f"[latex]{content}[/latex]"
        return placeholder
    
    content = re.sub(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', replace_inline_latex, content)
    
    return content, latex_blocks

def restore_latex(html_content, latex_blocks):
    """
    Restore LaTeX blocks in HTML content.
    
    Args:
        html_content (str): HTML content with LaTeX placeholders
        latex_blocks (dict): Dictionary of LaTeX blocks
        
    Returns:
        str: HTML content with LaTeX blocks restored
    """
    for placeholder, latex in latex_blocks.items():
        html_content = html_content.replace(placeholder, latex)
    
    return html_content

def markdown_to_wordpress(md_content):
    """
    Convert Markdown content to WordPress-compatible HTML.
    
    Args:
        md_content (str): Markdown content
        
    Returns:
        str: WordPress-compatible HTML content
    """
    # Preprocess LaTeX equations
    content, latex_blocks = preprocess_latex(md_content)
    
    # Process images
    content = process_images(content)
    
    # Convert Markdown to HTML
    md = Markdown(extensions=['extra', 'codehilite'])
    html_content = md.convert(content)
    
    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Process HTML elements
    soup = process_headings(soup)
    soup = process_lists(soup)
    soup = process_blockquotes(soup)
    soup = process_tables(soup)
    soup = process_links(soup)
    
    # Convert back to string
    html_content = str(soup)
    
    # Restore LaTeX blocks
    html_content = restore_latex(html_content, latex_blocks)
    
    # Final cleanup
    html_content = html_content.replace('\n\n', '\n')
    
    return html_content

def main():
    parser = argparse.ArgumentParser(description='Convert Markdown to WordPress HTML')
    parser.add_argument('input_file', help='Input Markdown file')
    parser.add_argument('output_file', help='Output WordPress HTML file')
    parser.add_argument('--title', help='Title for the WordPress post (optional)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        sys.exit(1)
    
    # Read input file
    with open(args.input_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert Markdown to WordPress HTML
    wp_content = markdown_to_wordpress(md_content)
    
    # Add title if provided
    if args.title:
        wp_content = f"{args.title}\n\n{wp_content}"
    
    # Write output file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(wp_content)
    
    print(f"Conversion complete: '{args.input_file}' -> '{args.output_file}'")

if __name__ == '__main__':
    main()
