import gradio as gr
import httpx
import json
from openai import OpenAI

# --- CONFIGURATION ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Replace with your OpenAI API Key
BACKEND_API_URL = "http://127.0.0.1:5000/analyse"

client = OpenAI(api_key=OPENAI_API_KEY)

def generate_ai_report(score: float, song_id: str):
    """Uses the OpenAI API to generate a qualitative report."""
    system_prompt = """
    You are an expert, encouraging, and friendly vocal coach. A student has just finished a practice session and received a quantitative pitch accuracy score. Your task is to provide a short, constructive, and motivating report based on this score.
    - If the score is high (above 90), praise their performance and consistency.
    - If the score is in the middle (70-90), encourage them, point out that they are on the right track, and suggest a small area for focus.
    - If the score is lower (below 70), be very encouraging. Acknowledge their effort and provide a simple, concrete, and positive tip for improvement, such as "a great next step is to practice humming the main melody slowly to solidify the notes in your mind."
    - Keep the tone supportive, not critical. Start with a positive opening.
    """
    user_prompt = f"The student practiced the song '{song_id}' and their pitch accuracy score is: {score} out of 100."
    try:
        response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating AI report: {e}"

# --- The main analysis function ---
async def analyze_singing(user_audio_file, song_id):
    """The main async generator function that Gradio will call."""
    if not user_audio_file:
        # --- FIXED LINE ---
        yield "Please upload an audio file.", ""
        return # Use a bare return to exit the generator

    yield "Uploading and processing...", ""
    
    try:
        files = {'user_audio': open(user_audio_file, 'rb')}
        data = {'song_id': song_id}
        
        async with httpx.AsyncClient(timeout=120.0) as client_http:
            response = await client_http.post(BACKEND_API_URL, files=files, data=data)
        
        response.raise_for_status()
        result = response.json()
        
        if "error" in result:
            # --- FIXED LINE ---
            yield f"Error from backend: {result['error']}", ""
            return

        score = result['score']
        
        yield f"Pitch score calculated: {score}%. Now generating AI feedback...", ""
        
        ai_feedback = generate_ai_report(score, song_id)

        final_report = f"## Vocal Analysis Report\n\n### Your Pitch Accuracy Score: **{score}%**\n\n---\n\n### AI Vocal Coach Feedback\n\n{ai_feedback}"
        
        # --- FIXED LINE ---
        yield "Analysis complete!", final_report

    except Exception as e:
        # --- FIXED LINE ---
        yield f"An error occurred: {e}", ""


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎤 Vocal Analyser")
    gr.Markdown("Upload your recording, select the song you sang, and get instant, AI-powered feedback!")
    
    with gr.Row():
        with gr.Column(scale=1):
            user_audio_input = gr.Audio(type="filepath", label="Upload Your Vocal Recording")
            song_id_input = gr.Dropdown(
                choices=[("shreya_ghoshal_agar_tum_mil_jao", "atj_sg")],
                value="atj_sg",
                label="Select Song"
            )
            submit_button = gr.Button("Analyze My Singing", variant="primary")
        with gr.Column(scale=20):
            status_output = gr.Textbox(label="Status", interactive=False)
            report_output = gr.Markdown(label="Your Report")
            
    submit_button.click(
        fn=analyze_singing,
        inputs=[user_audio_input, song_id_input],
        outputs=[status_output, report_output]
    )

if __name__ == "__main__":
    demo.launch()