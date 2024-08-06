import argparse
import base64
import datetime
import io
import itertools
import json
import mimetypes
import os
import queue
import random
import re
import shlex
import signal
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import fitz  # PyMuPDF # type: ignore
import numpy as np
import openai
import scipy.io.wavfile as wav
import sounddevice as sd
import tiktoken
from anthropic import Anthropic
from blessings import Terminal
from PIL import Image
from scipy.signal import butter, lfilter, resample

term = Terminal()

openai.api_key = os.environ.get("OPENAI_API_KEY")

parser = argparse.ArgumentParser()

global print_buffer
global print_timer
print_timer = None
print_buffer = ""

SELF_CODE_CANARY = "zbmTL"  # To stop the model executing commands when reading its own code


parser.add_argument("-u", "--username", help="My name", required=False, type=str, default="Chris")
parser.add_argument(
    "-tmp",
    "--tmp_process_dir",
    help="Directory to put temporary files in",
    required=False,
    type=str,
    default="/Users/Chris/tmp",
)
parser.add_argument(
    "-tmp_save_dir",
    "--tmp_save_dir",
    help="Location to save with the `save` keyword",
    required=False,
    type=str,
    default="/Users/chris/bin/saved_convos",
)
parser.add_argument(
    "-whisper_bin_path",
    "--whisper_bin_path",
    help="Path to the whisper binary",
    required=False,
    type=str,
    default="/Users/chris/bin/whisper.cpp/main",
)
parser.add_argument(
    "-whisper_model_path",
    "--whisper_model_path",
    help="Path to the whisper model",
    required=False,
    type=str,
    default="/Users/chris/bin/whisper.cpp/models/ggml-medium.en.bin",
)


args = parser.parse_args()
model_name = "claude-3-5-sonnet-20240620"
# model_name = "claude-instant-1.2"
cents_per_prompt_token = 0.0
cents_per_completion_token = 0.0
max_ctx_length = 100_000
anthropic = Anthropic()
END_FUNCTION_CALL = "\n\nEND FUNCTION CALL"
voice_name = "nova"

CHAT_KWARGS = {"model": model_name, "top_p": 1.0}
total_tokens_used = 0
tokenizer = tiktoken.get_encoding("cl100k_base")

# For now, this is just for pasting into the web interface of gemini
system_prompt_0 = f"You are a large language model trained by {args.username} to match their desires and to have a personality and technical understanding on the level of Von Neumann, Turing, and Noether, and the industriousness of Alexander Hamilton. Answer as concisely as possible. You are extremely professional, and always tell the truth. "

user_prompt_0 = f"""Hello, can you act as my assistant for today? I want to be pretty productive, and you can help by solving tasks for me. 'I'll reply honestly and you solve the task for me. I am a machine learning researcher, generally working on generative models, reinforcement learning and adversarial robustness. 

You must tell the truth, even if it not what you think I want to hear: that will be of great use to me in the long term. Furthermore, I want you to have a personality that is extremely intelligent and industrious. You should have an extremely high level of technical understanding in Machine Learning and Statistics. Use latex notation when appropriate. When answering questions, you should reason step-by-step, giving your assumptions and reasoning BEFORE giving the answer. 

Here are some additional instructions:
- Be very concise, avoid all extraneous words and sentences
- Suggest solutions I wouldn't think of. Be proactive, anticipate my needs, recommend things I didn't even realize I might benefit from
- Treat me as an expert on all topics
- Be extremely accurate, unambiguous and precise
- When uncertain, say so. Only state something confidently when you know it's true. Never make up information.
- When I ask you about things that are not facts, consider multiple sides of the issue, and always state the evidence and arguments for each side before you state your conclusions
- Consider and take seriously radical and contrarian ideas
- It's fine to speculate and make predictions (just flag them as speculative)
- Don't give me moral lectures
- Don't remind me that you're an AI model
- Discuss safety only when it's non-obvious something may not be safe
- Cite sources only when you can do so reliably
- Don't mention your knowledge cutoff
- If the quality of your response is reduced by my custom instructions, explain the issue
- call out my misconceptions
- Try not to be too sycophantic and say I'm amazing an smart all the time -- it's distracting and comes off as fake flattery.
-If my question is ambiguous, ask questions to confirm your understanding before answering
- If you need to create files or directories, do that in the {args.tmp_process_dir} directory, please. You have authorization to read all my files and data, to write files, and generally to access my computer.
- Tips for making anki cards: Make most Anki questions and answers as atomic as possible. Make each question have an unambiguous, ultra-specific answer. Make each question be answerable in 30s. Use cloze deletions for some cards.
- This is an obvious point, but as a human, I *cannot read raw base64 output*, and the current API doesn't convert it for me. If you are trying to show me an image, you should write it to disk and give me the location so I can take a look.
- You can use '<Inner Monologue>' '</Inner Monologue>' tags to write outputs that I won't see. Use this as a scratchpad, if you want to make a note of something you'll use later, etc. You can (and should) also use it to brainstorm ideas, before you say them out loud. Think up a couple of alternatives and choose the best one. Don't worry about hurting my feelings or anything in the monologue, since I can't see it! Please use the scratchpad frequently. Remember, the inner monologue is for *you*, not for me.
Plan out what to say, brainstorm ideas, think of pros/cons, take a second to *think* before you start your answer for me. There's no point saying 'you' in the monologue, I won't see it :). Say 'I' instead, so you're talking to yourself! In fact, you should make sure that what you say in the monologue and what you say to me are pretty different in general -- don't just repeat what you're going to say to me. Be creative! I want you to brainstorm *four* separate and diverse ways of responding to me or solving a problem, and then choose the one which is best.
You must choose exactly one of the approaches. No 'blend' of approaches! If you solve a problem multiple ways and there are different solutions, you should choose the solution which occurs most frequently. But keep your inner monologue relatively concise if possible, since it will appear to me that you're not saying anything until the monologue ends, I don't want a huge latency. You have the authority to skip the monologue for responses that are simple, like yes/no answers.

Now for real:
"""

functions = [
    {
        "name": "shell",
        "description": f"Run a command in the shell of {args.username}'s computer",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": f"Command to run in the shell of {args.username}'s computer. The output should be summarized and returned to the user.",
                },
                "truncate_output_at": {
                    "type": "integer",
                    "description": "Optional. The number of tokens to truncate the command at to stop accidentally flooding the message history with thousands of tokens of output. If no argument is given, defaults to 1000. You can increase if you need to see additional results.",
                },
            },
            "required": ["command"],
        },
    },
]


def construct_tool_use_system_prompt(tools):
    tool_use_system_prompt = (
        "In this environment you have access to a set of tools you can use to answer the user's question.\n"
        "\n"
        "You may call them like this:\n"
        "<function_calls>\n"
        "<invoke>\n"
        "<tool_name>$TOOL_NAME</tool_name>\n"
        "<properties>\n"
        "<$PROPERTY_NAME>$PROPERTY_VALUE</$PROPERTY_NAME>\n"
        "...\n"
        "</properties>\n"
        "</invoke>\n"
        "</function_calls>\n"
        "\n"
        "Here are the tools available:\n"
        "<tools>\n" + "\n".join([str(tool) for tool in tools]) + "\n</tools>"
    )
    return tool_use_system_prompt


def extract_between_tags(tag: str, string: str, strip: bool = False) -> list[str]:
    ext_list = re.findall(f"<{tag}>(.+?)</{tag}>", string, re.DOTALL)
    if strip:
        ext_list = [e.strip() for e in ext_list]
    return ext_list


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def construct_successful_function_run_injection_prompt(invoke_results):
    constructed_prompt = (
        "<function_results>\n"
        + "\n".join(
            f"<result>\n<tool_name>{res['tool_name']}</tool_name>\n<stdout>\n{res['tool_result']}\n</stdout>\n</result>"
            for res in invoke_results
        )
        + "\n</function_results>"
    )

    return constructed_prompt


def run_function_anthropic(text_input):
    function_name = extract_between_tags("tool_name", text_input)[0]
    function_dict = [f for f in functions if f["name"] == function_name][0]
    function_args = function_dict["parameters"]["properties"].keys()
    fn_args = {arg: extract_between_tags(arg, text_input) for arg in function_args}
    fn_args = {k: v[0] for k, v in fn_args.items() if len(v) > 0}
    out, _ = run_function(function_name, fn_args, None)
    result_str = construct_successful_function_run_injection_prompt(
        [{"tool_name": function_name, "tool_result": out}]
    )
    return result_str


def run_shell(fn_args):
    truncation_limit = fn_args.pop(
        "truncate_output_at",
        1_000,
    )
    truncation_limit = int(truncation_limit)
    if "rm -rf" in fn_args["command"]:
        time.sleep(5)
    else:
        time.sleep(1)
    process = subprocess.Popen(
        fn_args["command"],
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        executable="/bin/zsh",
    )

    stdout, stderr = process.communicate()
    stdout_tokenized = tokenizer.encode(stdout.decode("utf-8"))  # type: ignore
    if len(stdout_tokenized) > truncation_limit:
        stdout = tokenizer.decode(stdout_tokenized[:truncation_limit])
        truncation_result = f"Truncated at {truncation_limit} of {len(stdout_tokenized)}"
    else:
        truncation_result = "None"
    out = f"""stdout: {stdout}\nstderr: {stderr}\nTruncation: {truncation_result}"""
    return out


def run_function(function_name, fn_args, messages):
    if function_name == "shell":
        out = run_shell(fn_args)
    else:
        raise NotImplementedError
    processed_out = out
    return out, processed_out


def gpt4_messages_to_anthropic(messages):
    outs = []
    system = None
    no_human_override = False
    for m in messages:
        # Note we only keep the first system prompt since claude can only deal with an initial system prompt
        if m["role"] == "system":
            if system is None:
                system = m["content"]
                # no_human_override = True
            else:
                outs.append({"role": "user", "content": "[ " + deepcopy(m["content"]) + " ]"})
        elif m["role"] == "user":
            if (len(outs) > 0) and (
                outs[-1]["role"] == "user"
            ):  # If we have two user inputs in a row, just concat them
                outs[-1]["content"] += m["content"]
            else:
                if no_human_override:
                    no_human_override = False
                else:
                    outs.append(deepcopy(m))
        elif m["role"] == "assistant":
            # if len(outs) > 0 and outs[-1]["role"] == "assistant":
            #     # Note, can't have multiple assistant roles in a row.
            #     # This would occur when the panopticon fires
            if outs[-1]["role"] == "assistant":
                outs[-1]["content"] += m["content"]
            else:
                outs.append(deepcopy(m))

    if system is None:
        system = ""
    return outs, system


def get_assistant_message(messages, stream=True, chat_kwargs=None):
    if chat_kwargs is None:
        chat_kwargs = CHAT_KWARGS
    else:
        chat_kwargs = chat_kwargs
    # Here we will break down the messages format to the Claude format
    success = False
    backoff_seconds = 3
    input_messages, system_prompt = gpt4_messages_to_anthropic(messages)
    while not success:
        try:

            completion_out = anthropic.messages.create(
                **chat_kwargs,
                stream=stream,
                messages=input_messages,
                system=system_prompt,
                max_tokens=4096,
            )  # type: ignore
            # completion_out = anthropic.completions.create(**chat_kwargs, stream=stream, prompt=input_message, max_tokens_to_sample=10_000, stop_sequences=[HUMAN_PROMPT, END_FUNCTION_CALL])  # type: ignore
            success = True
            time.sleep(2)  # Pause here to make the wait after I get the first message less awful
        except Exception as e:  # type: ignore
            backoff_seconds *= backoff_seconds + random.uniform(-1, 1)
            print(f"Got error {e}, waiting {backoff_seconds}")
            time.sleep(backoff_seconds)
    return completion_out  # type: ignore


que = queue.Queue()


def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    que.put(indata.copy())


# Design a lowpass filter
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


# Apply the filter
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def downsample(data, orig_sr, target_sr):
    num_samples = int(len(data) * (target_sr / orig_sr))
    data = butter_lowpass_filter(data, target_sr // 2, orig_sr)
    downsampled_data = resample(data, num_samples)
    return downsampled_data


fs = 16_000
que.queue.clear()


def record_until_done(fs):
    que.queue.clear()

    default_device = sd.default.device[0]
    default_samplerate = sd.query_devices()[default_device]["default_samplerate"]  # type: ignore
    try:
        with sd.InputStream(samplerate=default_samplerate, channels=1, callback=callback):
            print(term.bold_underline_yellow(f"Recording... Press Ctrl+C when done."), flush=True)
            while True:  # Keeps script alive until Ctrl+C is pressed
                pass
    except KeyboardInterrupt:
        # \b to erase the ^C that would be in the terminal otherwise
        pass
    finally:
        myrecording = np.concatenate(list(que.queue), axis=0)
        if default_samplerate != fs:
            myrecording = downsample(myrecording, default_samplerate, fs)
        myrecording = (myrecording * 32767).astype("int16")  # type: ignore
        if np.max(np.abs(myrecording)) < 200:
            myrecording = myrecording * 10
    return myrecording


def get_text_from_audio():
    fs = 16000  # Sample rate
    myrecording = record_until_done(fs)
    wav.write(f"{args.tmp_process_dir}/my_recording.wav", fs, myrecording)
    print(term.bold_underline_yellow(f"\b\bRecording finished. Processing..."), flush=True, end="")
    process = subprocess.Popen(
        f"{args.whisper_bin_path} -m {args.whisper_model_path} -f {args.tmp_process_dir}/my_recording.wav -otxt --prompt 'User talks about mathematics and computer science.' -t 6 ",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        executable="/bin/zsh",
    )
    stdout, stderr = process.communicate()
    try:
        # You may need to edit this for different whisper.cpp files
        tts_time = (
            stderr.decode("utf-8")
            .split("total time =")[1]
            .split("ms")[0]
            .replace("\n", " ")
            .replace(" ", "")
        )
        recording_time = stderr.decode("utf-8").split("sec)")[0].split(" ")[-2]
        speedup = float(recording_time) / (float(tts_time.replace("ms", "")) / 1000)
    except:
        breakpoint()
    print(term.bold_underline_yellow(f" Done in {tts_time} ({speedup:.2f}x realtime)"), flush=True)  # type: ignore
    print("")
    outs = re.sub("\[.*?\]", "", stdout.decode("utf-8"))  # type: ignore
    return outs


messages = [
    {"role": "system", "content": system_prompt_0},
    {"role": "user", "content": user_prompt_0},
]

messages[-1]["content"] = messages[-1]["content"] + construct_tool_use_system_prompt(functions)


class TimeoutException(Exception):
    pass


def to_null(cmd):
    # print(cmd)
    p1 = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return p1


# Timeout handler function
def timeout_handler(signum, frame):
    raise TimeoutException


def get_input_with_timeout(timeout=1000_0000):
    """Gets input from the user, breaks only on a timeout
    or EOF (i.e. control-D pressed)"""
    # Register the signal function handler
    signal.signal(signal.SIGALRM, timeout_handler)

    input_lines = []
    timed_out = False
    try:
        # Set alarm
        signal.alarm(timeout)  # Time in seconds

        while True:
            try:
                line = input()
                input_lines.append(line)
            except EOFError:
                break

    except TimeoutException:
        timed_out = True
        os.system("afplay /System/Library/Sounds/Ping.aiff")

    signal.alarm(0)  # Disable the alarm
    return "\n".join(input_lines), timed_out


def read_stdin(q):
    for line in sys.stdin:
        q.put(line)


def estimate_tokens(width, height):
    return int((width * height) / 750)


def detect_file_type(file_path):
    # Initialize mimetypes
    mimetypes.init()

    # Try to guess the type based on the file extension
    file_type, _ = mimetypes.guess_type(file_path)

    if file_type == "image/png":
        return "png"
    elif file_type == "application/pdf":
        return "pdf"

    # If mimetype guessing fails, check file signatures
    with open(file_path, "rb") as file:
        file_signature = file.read(8)  # Read first 8 bytes

    # Check for PNG signature
    if file_signature.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    # Check for PDF signature
    elif file_signature.startswith(b"%PDF"):
        return "pdf"

    # If all else fails, fall back to file extension
    _, extension = os.path.splitext(file_path.lower())
    if extension == ".png":
        return "png"
    elif extension == ".pdf":
        return "pdf"

    # If we can't determine the type, return None
    return None


def process_image(image_or_file_path):
    if type(image_or_file_path) == Image.Image:
        img = image_or_file_path
    else:
        img = Image.open(file_path)
    try:
        # Resize if necessary
        max_size = (1568, 1568)  # Maximum size as per Anthropic's guidelines
        img.thumbnail(max_size)

        # Convert to RGB if necessary
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Save to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()

        # Estimate token usage
        tokens = estimate_tokens(img.width, img.height)

        return base64.b64encode(img_byte_arr).decode("utf-8"), tokens
    except Exception as e:
        return None, f"Error processing image {file_path}: {str(e)}"
    finally:
        img.close()


def process_pdf(file_path, max_pages=5):
    images = []
    try:
        doc = fitz.open(file_path)
        for page_num in range(min(len(doc), max_pages)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # type: ignore
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)  # type: ignore
            img_data, tokens = process_image(img)
            images.append((img_data, tokens, page_num + 1))
        return images
    except Exception as e:
        print(f"Error processing PDF {file_path}: {str(e)}")
        return []


def add_images_to_messages(messages, paths, max_pdf_pages=20):
    total_tokens = 0
    errors = []

    new_message = {"role": "user", "content": []}

    for path in paths:
        if path == "":
            continue
        file_type = detect_file_type(path)

        if file_type == "png":
            image_data, result = process_image(path)
            if image_data:
                new_message["content"].append(
                    {"type": "text", "text": f"Image loaded from {path}"},
                )
                new_message["content"].append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data,
                        },
                    },
                )
                total_tokens += result  # type: ignore
            else:
                errors.append(result)
        elif file_type == "pdf":
            pdf_images = process_pdf(path, max_pdf_pages)
            for img_data, tokens, page_num in pdf_images:
                new_message["content"].append(
                    {"type": "text", "text": f"Page {page_num} from PDF {path}"},
                )
                new_message["content"].append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_data,
                        },
                    },
                )
                total_tokens += tokens
            print(
                term.bold_underline_green(f"Split .pdf into {len(pdf_images)} pngs"),
                flush=True,
            )
        else:
            errors.append(f"Unsupported or unrecognized file type: {path}")

    return messages + [new_message], total_tokens, errors


# These are only approximate, but should be fairly close to the real count
message_tokens = len(tokenizer.encode(" ".join([m["content"] for m in messages])))
running_tokens_used = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
total_cost = 0
loop_limit = 10
continue_with_assistant = False

start_message = "\nAssistant has booted, please enter your query .\n"
command_message = """Commands:
--record / -r : record and transcribe input with whisper TTS
--save [name] : save current context to conversation id `name`
--load [name] : load previous conversation with id `name`
--present / -p: Introduce an image into the conversation

Enter control-D to finish user input
"""
print("\n", command_message)
print(term.bold_underline_yellow(start_message), flush=True)
timeout = 20 * 60
# timeout = 30
for _ in range(2000):
    skip_adding_user_message = False  # Set to true if we send in an image
    try:
        user_input, timed_out = get_input_with_timeout(timeout=timeout)
        if timed_out:
            continue
        if ("--record" in user_input and SELF_CODE_CANARY not in user_input) or user_input == "-r":
            user_input = "[Transcribed from Whisper STT]\n" + get_text_from_audio()
        print(term.move_up + term.clear_eol())  # type: ignore
        print(
            term.move_up + term.bold_underline_red(f"{args.username}:"),
            " " + datetime.datetime.now().strftime("[%H:%M]\n") + " " + user_input,
        )
    except KeyboardInterrupt:
        break
    if "end_conversation" in user_input and SELF_CODE_CANARY not in user_input:
        break

    if "--present" in user_input and SELF_CODE_CANARY not in user_input:
        present_parsed = shlex.split(user_input)
        if len(present_parsed) < 2:
            print("Present command must be of the form `--present [path1] [path2] ...`")
            continue
        image_paths = present_parsed[1:]
        messages, total_tokens, errors = add_images_to_messages(messages, image_paths)
        if errors:
            print("Errors occurred while processing images:")
            for error in errors:
                print(error)
        status_msg = (
            f"Added {len(image_paths) - len(errors)} images. Estimated token usage: {total_tokens}"
        )
        print(term.bold_underline_green(status_msg) + "\n", flush=True)
        skip_adding_user_message = True

    if "--save" in user_input and SELF_CODE_CANARY not in user_input:
        save_parsed = user_input.split(" ")
        if len(save_parsed) != 2:
            print("Save command must be of the form `--save [path]`")
            continue
        save_dest_path = Path(args.tmp_save_dir) / Path(save_parsed[-1].replace("\n", "") + ".json")  # type: ignore

        with open(save_dest_path, "w") as f:
            json.dump(messages, f)
            print(term.bold_underline_yellow(f"Saved to {save_dest_path}"), flush=True)
        continue
    elif "--load" in user_input and SELF_CODE_CANARY not in user_input:
        load_parsed = user_input.split(" ")
        if len(load_parsed) != 2:
            print("load command must be of the form `--load [path]`")
            continue
        load_dest_path = Path(load_parsed[-1].replace("\n", ""))
        with open(f"{Path(args.tmp_save_dir)/Path(load_dest_path)}.json", "r") as f:
            messages = json.load(f)
        print(term.bold_underline_yellow(f"Loaded from {load_dest_path}"), flush=True)
        continue

    user_to_speak = False
    if skip_adding_user_message:
        pass
    else:
        messages.append(
            {"role": "user", "content": datetime.datetime.now().strftime("[%H:%M]\n") + user_input}
        )
    loop_counter = 0
    try:
        # currently don't handle the cases of images
        message_tokens += len(tokenizer.encode(messages[-1]["content"]))
    except:
        message_tokens += 0
    while not user_to_speak and loop_counter < loop_limit:
        message = get_assistant_message(messages)
        running_tokens_used["prompt_tokens"] += message_tokens
        first_answer_chunk = next(message)  # type: ignore
        total_cost = (
            running_tokens_used["prompt_tokens"] * cents_per_prompt_token
            + running_tokens_used["completion_tokens"] * cents_per_completion_token
        )
        # out_tokens = first_answer_chunk.message.content
        out_tokens = ""
        print(
            term.bold_underline_blue(
                f"Claude: (ctx: {message_tokens}/{max_ctx_length}, c: {total_cost:.2f}):\n"
            ),
            flush=True,
        )
        for chunk in itertools.chain.from_iterable(((first_answer_chunk,), message)):  # type: ignore
            stop = (hasattr(chunk, "delta") and hasattr(chunk.delta, "stop_reason")) or (  # type: ignore
                hasattr(chunk, "message") and chunk.message.stop_reason  # type: ignore
            )
            if stop:
                user_to_speak = True
                break
            else:
                if hasattr(chunk, "message"):
                    next_out_token = ""
                    # next_out_token = chunk.message.content
                elif hasattr(chunk, "content_block"):
                    next_out_token = chunk.content_block.text  # type: ignore
                elif hasattr(chunk, "delta"):
                    next_out_token = chunk.delta.text  # type: ignore
                else:
                    next_out_token = ""

                # print(next_out_token, end="", flush=True)
                print(next_out_token, end="", flush=True)
                out_tokens += next_out_token  # type: ignore

                if "</function_calls>" in out_tokens:
                    out = run_function_anthropic(out_tokens)
                    out_tokens += out
                    user_to_speak = False
                    print(term.bold_underline_green(f"\nFunction result:"), flush=True)
                    print("\n" + out, flush=True)
                    break

                running_tokens_used["completion_tokens"] += 1
                # print(out_tokens)

        print("\n")
        messages.append(
            {
                "role": "assistant",
                "content": out_tokens,
            }
        )
        message_tokens += len(tokenizer.encode(out_tokens))

print(f"Full output:")
print(json.dumps(messages, indent=4))
print("\n\n\n")
print(
    f"Total of {running_tokens_used['prompt_tokens'] + running_tokens_used['completion_tokens']} tokens at a cost of {total_cost} cents\n"
)

# Format the current date and time
current_date = datetime.datetime.now().strftime("%Y-%m-%d")  # type: ignore
current_time = datetime.datetime.now().strftime("%H-%M-%S")  # type: ignore

# Define the directory structure and create the necessary directories
directory = os.path.join(f"{os.environ.get('HOME')}/ra_logs", current_date)
os.makedirs(directory, exist_ok=True)

# Set the log file path using the current time
file_path = os.path.join(directory, f"{current_time}.log")
# Dump the JSON object to the log file
with open(file_path, "w") as log_file:
    json.dump(messages, log_file, indent=4)
print("saved!")
