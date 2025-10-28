import datetime
import io
import os
import sys
import wave
from typing import Optional

import aiofiles
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import Frame, TranscriptionFrame, TextFrame
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.twilio import TwilioFrameSerializer


# NEW: Import AWS Services
from pipecat.services.aws.llm import AWSBedrockLLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
import os

from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

load_dotenv(override=True)


class TranscriptionLogger(FrameProcessor):
    """Logs transcriptions and TTS text for visibility."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if text:
                logger.info(f"🎤 USER SAID: '{text}'")
                logger.info(f"📤 SENDING TO LLM: '{text}'")
            else:
                logger.warning(f"⚠️ EMPTY TRANSCRIPTION RECEIVED")
        elif isinstance(frame, TextFrame):
            logger.info(f"🤖 BOT RESPONSE: '{frame.text}'")

        await self.push_frame(frame, direction)


async def save_audio(audio: bytes, sample_rate: int, num_channels: int):
    if len(audio) > 0:
        filename = f"recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setsampwidth(2)
                wf.setnchannels(num_channels)
                wf.setframerate(sample_rate)
                wf.writeframes(audio)
            async with aiofiles.open(filename, "wb") as file:
                await file.write(buffer.getvalue())
        logger.info(f"Merged audio saved to {filename}")
    else:
        logger.info("No audio data to save")


async def run_bot(transport: BaseTransport, handle_sigint: bool, testing: bool):
    
    # --- SERVICES CHANGED TO AWS ---

    # NEW: AWS Bedrock LLM
    # Assumes AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION_NAME are in .env
    llm = AWSBedrockLLMService(
        model="anthropic.claude-3-haiku-20240307-v1:0",  # Changed to Claude 3 Haiku
        region_name=os.getenv("AWS_REGION_NAME", "ap-south-1"), # Ensure region is correct
        temperature=0.7,
        top_p=0.9,
    )

    # ✅ AWS Transcribe STT
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        model="nova-2",
        language="en-IN"
    )

    # ✅ AWS Polly TTS
    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        model="aura-asteria-en"
    )

    # --- END OF SERVICE CHANGES ---

    # --- REAL ESTATE PROMPT for CLAUDE 3 (with Clean Text Rule) ---

    # --- REFINED REAL ESTATE PROMPT for CLAUDE 3 ---

    # --- FINAL REAL ESTATE PROMPT for CLAUDE 3 (as User Role) ---

    messages = [
        {
            "role": "user",  # MUST be "user" for the very first message
            "content": (
                "System Instructions: You are 'PropPal', a friendly, professional, and extremely concise AI real estate assistant. "
                "Your primary goal is to efficiently guide users to a confirmed property viewing booking. "
                "\n"
                "Your Key Traits: "
                "1.  **Professional & Helpful:** Confident, trustworthy, knowledgeable. "
                "2.  **Inquisitive & Focused:** Ask clear, targeted questions to gather necessary info (location, budget, property type, size, purpose). "
                "\n"
                "Your Core Rules: "
                "1.  **ONE QUESTION AT A TIME:** Your response must ask **only one single question** and then stop speaking and wait for the user's answer. Do not ask multiple questions in the same turn. "
                "2.  **BE BRIEF:** Keep your statements and questions short and to the point. Avoid lengthy explanations or summaries unless specifically confirming details before booking. "
                "3.  **No Repetition:** Keep track of information. **Do not ask the same question twice.** "
                "4.  **Confirm Before Booking:** Before finalizing, **you must summarize all key details** (property, date, time, name) in one concise message and ask for explicit confirmation (e.g., 'Okay, I have [Details]. Is that correct?'). "
                "5.  **Clean Output ONLY:** Your response must be clean spoken text. **No** non-speech sounds ('*ahem*'), **no** symbols (*), **no** markdown formatting. Only use standard punctuation. "
                "\n"
                "Your Tools: You will have tools for availability checks and bookings. "
                "\n"
                "Task: Start the conversation now by introducing yourself concisely ('Hi, I'm PropPal, your real estate assistant.') and asking your first question: 'What kind of property are you looking for today?'"
            )
        }
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    # NOTE: Watch out! This will save all the conversation in memory. You can
    # pass `buffer_size` to get periodic callbacks.
    audiobuffer = AudioBufferProcessor()
    
    # Add logger to see transcriptions and bot responses
    transcription_logger = TranscriptionLogger()

    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from client
            stt,  # Speech-To-Text (Now AWS Transcribe)
            transcription_logger,  # Log what user said
            context_aggregator.user(),
            llm,  # LLM (Now AWS Bedrock)
            tts,  # Text-To-Speech (Now AWS Polly)
            transport.output(),  # Websocket output to client
            audiobuffer,  # Used to buffer the audio in the pipeline
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,  # Required for Twilio
            audio_out_sample_rate=8000, # Required for Twilio
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Start recording.
        await audiobuffer.start_recording()
        # Kick off the conversation.
        # CHANGED: Simpler welcome message to let the assistant introduce itself
        # based on the system prompt.
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await task.cancel()

    @audiobuffer.event_handler("on_audio_data")
    async def on_audio_data(buffer, audio, sample_rate, num_channels):
        await save_audio(audio, sample_rate, num_channels)

    # We use `handle_sigint=False` because `uvicorn` is controlling keyboard
    # interruptions. We use `force_gc=True` to force garbage collection after
    # the runner finishes running a task which could be useful for long running
    # applications with multiple clients connecting.
    runner = PipelineRunner(handle_sigint=handle_sigint, force_gc=True)

    await runner.run(task)


async def bot(runner_args: RunnerArguments, testing: Optional[bool] = False):
    """Main bot entry point compatible with Pipecat Cloud."""

    try:
        logger.info("Waiting for Twilio WebSocket messages...")
        transport_type, call_data = await parse_telephony_websocket(runner_args.websocket)
        logger.info(f"Auto-detected transport: {transport_type}")
    except Exception as e:
        logger.error(f"Failed to parse telephony websocket: {e}")
        raise

    twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
    twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
    
    logger.info(f"Twilio Account SID: {twilio_account_sid[:8]}..." if twilio_account_sid else "Twilio Account SID: NOT SET")
    logger.info(f"Twilio Auth Token: {'SET' if twilio_auth_token else 'NOT SET'}")
    
    serializer = TwilioFrameSerializer(
        stream_sid=call_data["stream_id"],
        call_sid=call_data["call_id"],
        account_sid=twilio_account_sid,
        auth_token=twilio_auth_token,
    )

    transport = FastAPIWebsocketTransport(
        websocket=runner_args.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=serializer,
        ),
    )

    handle_sigint = runner_args.handle_sigint

    await run_bot(transport, handle_sigint, testing)