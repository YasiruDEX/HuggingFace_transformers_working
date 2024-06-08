from ctransformers import AutoModelForCausalLM

# Initialize the model
llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-Instruct-v0.1-GGUF", 
    model_file="mistral-7b-instruct-v0.1.Q4_K_S.gguf", 
    model_type="mistral", 
    gpu_layers=0
)

# Function to generate a response from the model
def generate_response(system_prompt, user_prompt):
    # Combine the system and user prompts into a single input
    input_text = f"System: {system_prompt}\nUser: {user_prompt}\nAI:"
    # Generate response
    response = llm(input_text)
    return response

# Define the system prompt
system_prompt = """You are a prompt generator for a stable diffusion model. You have to extract important facial features from a given description give a comma-separated string.
                # Example - Suspect is a 30 years old Caucasian with neck-length short blond hair. She has a round face with a big forehead and a round chin. Her eyes are very thin and small, and her mouth is small as well. She has filled cheeks and a thin nose.
                # Output - 30 years old Caucasian woman, neck length short blond hair, round face, very thin small eyes, round chin, small mouth, big forehead, filled cheeks, thin nose

                # Example - Suspect is a 40-year-old East Asian man. He has straight, shoulder-length black hair neatly combed. His face is oval-shaped with high cheekbones and a defined jawline. He possesses almond-shaped brown eyes . His nose is straight and of average size.
                # Output - 40 years old East Asian man, shoulder length black hair, neatly combed hair, oval face, almond-shaped brown eyes, high cheekbones, defined jawline, medium-sized mouth, full lips, straight nose
                """

# Main loop to interact with the user
while True:
    user_prompt = input("User: ")
    if user_prompt.lower() in ["exit", "quit", "stop"]:
        print("Exiting the chat. Goodbye!")
        break
    response = generate_response(system_prompt, user_prompt)
    print(f"AI: {response}")

