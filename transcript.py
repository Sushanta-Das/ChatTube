from langchain_community.document_loaders.parsers.audio import AzureOpenAIWhisperParser
import os
import gc
import time
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound,TranscriptsDisabled,CouldNotRetrieveTranscript
from langchain_community.document_loaders.blob_loaders.youtube_audio import (
    YoutubeAudioLoader,
)
from langchain_community.document_loaders.generic import GenericLoader
load_dotenv()  # Load variables from .env






def download_and_generate_transcript(url):
    endpoint = os.environ["AZURE_WHISPER_ENDPOINT"] 
    key = os.environ["AZURE_WHISPER_API_KEY"] 
    version =os.environ["AZURE_WHISPER_VERSION"] 
    name = "whisper"
    parser = AzureOpenAIWhisperParser(
        api_key=key, azure_endpoint=endpoint, api_version=version, deployment_name=name
    )
    url=[url]
    save_dir = os.path.join(os.getcwd(), "downloads")
    loader = GenericLoader(
    YoutubeAudioLoader(url, save_dir), parser
    )

    docs = loader.load()
    transcript=" ".join([doc.page_content for doc in docs])
    #delete the downloaded files
    # Ensure resources are cleaned up before deleting files
    del loader
    del parser
    gc.collect()
    time.sleep(1)  # allow file handles to be released

    # Delete the downloaded files
    for doc in docs:
        source_path = doc.metadata.get("source")
        if source_path and os.path.exists(source_path):
            try:
                os.remove(source_path)
            except PermissionError:
                print(f"Could not delete file: {source_path} — still in use.")
    return transcript    


# ---------------------- Utilities ----------------------
def extract_video_id(url):
    parsed = urlparse(url)
    if parsed.hostname in ["www.youtube.com", "youtube.com"]:
        return parse_qs(parsed.query).get("v", [None])[0]
    elif parsed.hostname == "youtu.be":
        return parsed.path.lstrip("/")
    return None


def get_transcript(video_url,preferred_lang="en"):
      video_id = extract_video_id(video_url)
      try:
        # Try manually uploaded transcript in preferred language
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[preferred_lang])
        return " ".join([entry['text'] for entry in transcript])
      except (NoTranscriptFound, TranscriptsDisabled):
        try:
            # Get available transcripts (manual + auto)
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            # Try auto-generated transcript in preferred language
            try:
                auto_transcript = transcript_list.find_generated_transcript([preferred_lang])
                return " ".join([entry['text'] for entry in auto_transcript.fetch()])
            except NoTranscriptFound:
                # Fallback: get any available transcript (auto or manual)
                for transcript in transcript_list:
                    try:
                        return " ".join ( item.text for item in  transcript.fetch())
                    except CouldNotRetrieveTranscript:
                        continue
                return None
        except Exception as e:
            # if transcript is not available in any language, then generate transcript by downloading the video
            # and using a speech-to-text model (e.g., Whisper)
        
            print(f"Transcript error: {e}")
            print("Downloading video for transcript generation...")
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            transcript = download_and_generate_transcript(video_url)
            if transcript:
                return transcript
            else:
                print("Transcript generation failed.")
                return None
                
if __name__ == "__main__":
    # Example usage
    url = "https://www.youtube.com/watch?v=BnEK7OkXA-4"
    video_id = extract_video_id(url)
    if video_id:
        transcript = download_and_generate_transcript(url)
        if transcript:
            print("Transcript:", transcript)
        else:
            print("No transcript available.")
    else:
        print("Invalid YouTube URL.")