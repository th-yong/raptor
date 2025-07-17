from chunking_evaluation.chunking import ClusterSemanticChunker
from chunking_evaluation.utils import get_openai_embedding_function, openai_token_count
from langchain_openai import AzureOpenAIEmbeddings
import re
import tiktoken
import os
from dotenv import load_dotenv

load_dotenv()

class ClusterSemanticTableTextSplitter:
    def __init__(self,  max_chunk_size=512, min_chunk_size=100):
        self.table_pattern = re.compile(
            r'--- \[Table (\d+) Start\] ---.*?--- \[Table Summary \1 End\] ---',
            re.DOTALL
        )
        self.table_summary_pattern = re.compile(
            r'--- \[Table Summary \d+ Start\] ---\s*(.*?)\s*--- \[Table Summary \d+ End\] ---',
            re.DOTALL
        )
        self.page_marker_pattern = re.compile(r'--- Page (\d+) ---')
        azure_embedder = AzureOpenAIEmbeddings(model="text-embedding-3-large",
                                               openai_api_key = os.getenv("AZURE_OPENAI_KEY"),
                                               azure_endpoint= os.getenv("AZURE_OPENAI_ENDPOINT"),
                                               )

        def embedding_function(batch: list[str]) -> list[list[float]]:
            return azure_embedder.embed_documents(batch)
        
        self.chunker = ClusterSemanticChunker(
            embedding_function=embedding_function,
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
            length_function=openai_token_count
        )


    def split_documents(self, texts: list[str]) -> list[str]:
        all_chunks = []

        for doc in texts:
            text = doc
            page_map = self._get_page_map_from_raw_text(text)
            last = 0

            for match in self.table_pattern.finditer(text):
                table_start, table_end = match.span()
                table_block = match.group(0)
                table_summary = self._extract_summary(table_block)

                if last < table_start:
                    pre_text = text[last:table_start]
                    pre_chunks = self._cluster_split(
                        pre_text, page_map, last,
                        table_block=table_block,
                        table_summary=table_summary
                    )
                    all_chunks.extend(pre_chunks)

                last = table_end

            if last < len(text):
                post_text = text[last:]
                post_chunks = self._cluster_split(post_text, page_map, last)
                all_chunks.extend(post_chunks)

        return all_chunks

    def _cluster_split(
        self, text: str, page_map: list[tuple[int, int]], offset: int,
        table_block: str = None, table_summary: str = ""
    ) -> list[str]:
        base_chunks = self.chunker.split_text(text)
        results = []
        table_inserted = False

        for chunk in base_chunks:
            abs_start = offset + text.find(chunk)
            page = self._find_page_for_index(abs_start, page_map)

            is_table = False
            table_content = ""
            table_content_summary = ""

            if table_block and not table_inserted:
                table_body_only = self._clean_text(self._extract_table_only(table_block))
                table_summary_clean = self._clean_text(table_summary)

                chunk += "\n\n" + table_body_only
                is_table = True
                table_content = table_body_only
                table_content_summary = table_summary_clean
                table_inserted = True

            chunk_text = self._clean_text(chunk)
            if not chunk_text.strip() or len(chunk_text) < 10:
                continue
            # results.append({
            #     "doc_type" : "other_terms",
            #     "content" : chunk_text,
            #     "page_number": page,
            #     "is_table": is_table,
            #     "table_content": table_content,
            #     "table_content_summary": table_content_summary,
            # })
            results.append(f"{chunk_text} {table_content} {table_content_summary}")
        return results

    def _clean_text(self, text: str) -> str:
        patterns = [
            r'--- Page \d+ ---',
            r'--- \[Table \d+ Start\] ---',
            r'--- \[Table \d+ End\] ---',
            r'--- \[Table Summary \d+ Start\] ---',
            r'--- \[Table Summary \d+ End\] ---'
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text)
        return text.strip()

    def _extract_summary(self, table_text: str) -> str:
        match = self.table_summary_pattern.search(table_text)
        return match.group(1).strip() if match else ""

    def _extract_table_only(self, table_block: str) -> str:
        summary_match = re.search(
            r'--- \[Table Summary \d+ Start\] ---', table_block
        )
        if summary_match:
            return table_block[:summary_match.start()].strip()
        return table_block.strip()

    def _get_page_map_from_raw_text(self, text: str) -> list[tuple[int, int]]:
        return [(int(m.group(1)), m.start()) for m in self.page_marker_pattern.finditer(text)]

    def _find_page_for_index(self, index: int, page_map: list[tuple[int, int]]) -> int:
        for page_num, start in reversed(page_map):
            if index >= start:
                return page_num
        return page_map[0][0] if page_map else 1

if __name__ == "__main__":
    
    filename = "현대해상3(퇴직연금)상품약관"
    
    with open(f"../demo/{filename}.txt", "r", encoding="utf-8") as f:
        content = f.read()
        
    # doc = Document(page_content=content)
    splitter = ClusterSemanticTableTextSplitter(max_chunk_size=512, min_chunk_size=100)
    chunks = splitter.split_documents([content])
 