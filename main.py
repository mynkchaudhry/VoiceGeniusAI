import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import requests
import tempfile
import pymupdf4llm
import json
from dotenv import load_dotenv
import logging
from openai import OpenAI
from pymongo import MongoClient
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware


# Set up logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("pymongo").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

origins = [
    "http://localhost:8000",  # React/Vue/Next.js running locally
    "http://127.0.0.1:5500",
    "https://voicegeniusai.onrender.com/"
]

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize MongoDB client with enhanced SSL/TLS options
try:
    mongo_client = MongoClient(
        MONGODB_URI,
        tls=True,
        tlsAllowInvalidCertificates=False,  # Ensure valid certificates
        retryWrites=True,
        serverSelectionTimeoutMS=30000
    )
    # Test the connection
    mongo_client.admin.command('ping')
    logger.info("Successfully connected to MongoDB")
    
    db = mongo_client["persona_generator"]
    personas_collection = db["personas"]
    counters_collection = db["counters"]
    
    # Initialize counter if it doesn't exist
    if counters_collection.count_documents({"_id": "train_id"}) == 0:
        counters_collection.insert_one({"_id": "train_id", "sequence_value": 0})
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")
    # Continue execution, but log the error

app = FastAPI(title="Outbound Call Persona Generator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Specifies which domains can access your API
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)


# Updated Pydantic model for the response format
class PersonaResponse(BaseModel):
    train_id: int
    prompt: str
    created_at: datetime

class PersonaCustomization(BaseModel):
    train_id: int
    name: str
    gender: str
    language: str
    role: str

def get_next_sequence_value(sequence_name):
    """Get the next value in the specified sequence"""
    try:
        sequence_document = counters_collection.find_one_and_update(
            {"_id": sequence_name},
            {"$inc": {"sequence_value": 1}},
            return_document=True
        )
        return sequence_document["sequence_value"]
    except Exception as e:
        logger.error(f"Failed to get next sequence value: {str(e)}")
        # Fallback to a timestamp-based ID if MongoDB is unavailable
        return int(datetime.utcnow().timestamp())

@app.post("/generate-persona", response_model=PersonaResponse)
async def generate_persona(
    file: UploadFile = File(...)
):
    """
    Generate a personalized outbound call system prompt and script based on input.
    
    - Accepts audio (mp3, wav), PDF, or text files
    - Analyzes file content based on file type
    - Creates a comprehensive prompt with both system instructions and example script
    - Returns in a format ready for use with LLM-powered calling systems
    - Stores the response in MongoDB with a unique auto-incrementing train_id
    """
    try:
        logger.debug(f"Processing file: {file.filename}")
        
        extracted_text = ""
        
        # Process the file
        file_content = await file.read()
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            # Process based on file type
            if file_extension in ['.mp3', '.wav', '.m4a', '.flac']:
                # Audio transcription using OpenAI's Whisper model
                logger.debug(f"Processing audio file: {file_extension}")
                extracted_text = transcribe_audio(temp_file_path)
            
            elif file_extension == '.pdf':
                # PDF extraction using PyMuPDF
                logger.debug("Processing PDF file")
                extracted_text = extract_text_from_pdf(temp_file_path)
            
            elif file_extension in ['.txt', '.doc', '.docx']:
                # Text file extraction
                logger.debug(f"Processing text file: {file_extension}")
                with open(temp_file_path, 'r', encoding='utf-8') as f:
                    extracted_text = f.read()
            
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_extension}")
                
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
        
        # Generate persona with OpenAI GPT-4o-mini
        logger.debug("Generating persona with OpenAI GPT-4o-mini")
        persona_text = generate_persona_with_openai(extracted_text)
        
        # Get next train_id
        train_id = get_next_sequence_value("train_id")
        
        # Create response with timestamp
        timestamp = datetime.utcnow()
        response_data = {
            "train_id": train_id,
            "prompt": persona_text["prompt"],
            "created_at": timestamp,
            "filename": file.filename,
            "file_type": file_extension
        }
        
        # Store in MongoDB
        try:
            personas_collection.insert_one(response_data)
            logger.debug(f"Stored persona in MongoDB with train_id: {train_id}")
        except Exception as e:
            logger.error(f"Failed to store in MongoDB: {str(e)}")
            # Continue execution even if MongoDB storage fails
        
        # Return response
        return PersonaResponse(
            train_id=train_id,
            prompt=persona_text["prompt"],
            created_at=timestamp
        )
    except Exception as e:
        logger.exception("Unhandled exception in generate_persona endpoint")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

def transcribe_audio(file_path):
    """Transcribe audio file using OpenAI's Whisper model"""
    logger.debug(f"Attempting to transcribe audio file: {file_path}")
    
    try:
        with open(file_path, "rb") as audio_file:
            logger.debug("Sending request to OpenAI API for audio transcription")
            
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        
        logger.debug(f"Successfully transcribed audio: {transcription.text[:100]}...")
        return transcription.text
    except Exception as e:
        logger.error(f"Audio transcription failed: {str(e)}")
        raise HTTPException(status_code=500, 
                           detail=f"Audio transcription failed: {str(e)}")

def extract_text_from_pdf(file_path):
    """Extract text from PDF using PyMuPDF"""
    try:
        logger.debug(f"Extracting text from PDF: {file_path}")
        text = pymupdf4llm.to_markdown(file_path)
        logger.debug(f"Successfully extracted PDF text: {text[:100]}...")
        return text
    except Exception as e:
        logger.exception("PDF extraction failed")
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {str(e)}")

def generate_persona_with_openai(content):
    """Generate a persona using OpenAI's GPT-4o-mini"""
    logger.debug("Constructing prompt for OpenAI GPT-4o-mini")
    
    # Construct the prompt
    system_message = """
    You are an expert at creating persuasive outbound call personas and scripts.
    Your task is to create a single comprehensive prompt that includes:
    
    1. A detailed system prompt section that defines a professional role and objectives 
    2. A complete call script example with specific sections
    3. Additional system guidance for the caller
    
    The prompt should be generic (no specific caller names) and in a format that can be directly used.
    """
    
    user_message = f"""
    Create a comprehensive outbound caller prompt based on the provided content.
    The prompt should be formatted as follows:
    
    **System Prompt:**
    [Define the caller's role at the specified company without using specific names. Include what they do, their expertise, and their main purpose for calling.]
    
    **When engaging a lead, your main objectives are to:**
    - [List 4-5 key objectives with each one in bold]
    
    [Additional context about how to handle the call professionally]
    
    ---
    **Call Script Example:**
    1. *Opening the Call:*
       - [Example greeting without specific names]
       - [Purpose statement]
       
    2. *Qualifying the Lead:*
       - [2-3 example questions]
       
    3. *Pitching the Service:*
       - [How to present the solution based on prospect's needs]
       
    4. *Handling Objections:*
       - [2-3 examples of handling common objections]
       
    5. *Closing the Deal:*
       - [1-2 examples of how to set next steps]
       
    *End the call by:*
       - [Example of wrap-up]
    
    ---
    **System Guidance:**
    - [3-4 bullet points with additional tips for the caller]
    ---
    
    Here is the content to base the persona and script on:
    
    {content}
    """
    
    try:
        logger.debug("Sending request to OpenAI API for GPT-4o-mini generation")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7
        )
        
        logger.debug("Successfully received OpenAI response")
        
        formatted_prompt = response.choices[0].message.content.strip()
        
        return {"prompt": formatted_prompt}
    except Exception as e:
        logger.exception("Failed to generate OpenAI response")
        raise HTTPException(status_code=500, 
                           detail=f"Failed to generate OpenAI response: {str(e)}")

@app.post("/customize-persona", response_model=PersonaResponse)
async def customize_persona(customization: PersonaCustomization):
    """
    Customize an existing persona by appending gender, language, name, and role information.
    
    - Takes a train_id and customization parameters
    - Retrieves the existing prompt associated with the train_id
    - Appends the customization information to the prompt
    - Updates the stored prompt in MongoDB
    - Returns the updated persona
    """
    try:
        logger.debug(f"Customizing persona with train_id: {customization.train_id}")
        
        # Retrieve the existing persona
        try:
            existing_persona = personas_collection.find_one({"train_id": customization.train_id})
            
            if not existing_persona:
                raise HTTPException(status_code=404, detail=f"Persona with train_id {customization.train_id} not found")
            
            existing_prompt = existing_persona.get("prompt", "")
        except Exception as e:
            logger.error(f"Failed to retrieve persona from MongoDB: {str(e)}")
            # Create a fallback persona if MongoDB retrieval fails
            existing_prompt = "Fallback prompt due to database connectivity issues."
            existing_persona = {"created_at": datetime.utcnow()}
        
        # Create customization text to append
        customization_text = f"""
        ---
        **Some Information about you:**
        - **You are {customization.role} and your name is: {customization.name}.**
        - **You are a {customization.gender}.**
        - **You speak {customization.language}, so you will only respond in this language.**
        ---
        """
        
        # Append customization to the existing prompt
        updated_prompt = existing_prompt + "\n" + customization_text
        
        # Update in MongoDB
        try:
            personas_collection.update_one(
                {"train_id": customization.train_id},
                {"$set": {"prompt": updated_prompt, "customized_at": datetime.utcnow(),
                        "customization": {
                            "name": customization.name,
                            "gender": customization.gender,
                            "language": customization.language,
                            "role": customization.role
                        }}}
            )
            logger.debug(f"Updated persona with train_id: {customization.train_id}")
        except Exception as e:
            logger.error(f"Failed to update persona in MongoDB: {str(e)}")
            # Continue execution even if MongoDB update fails
        
        # Return updated persona
        return PersonaResponse(
            train_id=customization.train_id,
            prompt=updated_prompt,
            created_at=existing_persona.get("created_at", datetime.utcnow())
        )
    
    except Exception as e:
        logger.exception("Unhandled exception in customize_persona endpoint")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Helper endpoints for testing and management

@app.get("/personas/{train_id}")
async def get_persona(train_id: int):
    """Retrieve a persona by train_id"""
    try:
        persona = personas_collection.find_one({"train_id": train_id})
        if not persona:
            raise HTTPException(status_code=404, detail="Persona not found")
        
        # Convert ObjectId to string for JSON serialization
        persona["_id"] = str(persona["_id"])
        return persona
    except Exception as e:
        logger.error(f"Failed to retrieve persona: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/personas")
async def list_personas(limit: int = 10, skip: int = 0):
    """List personas with pagination"""
    try:
        personas = list(personas_collection.find().sort("train_id", -1).skip(skip).limit(limit))
        # Convert ObjectId to string for JSON serialization
        for persona in personas:
            persona["_id"] = str(persona["_id"])
        
        return {"personas": personas, "total": personas_collection.count_documents({})}
    except Exception as e:
        logger.error(f"Failed to list personas: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# Root endpoint for health check
@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "online", "message": "API is running"}

if __name__ == "__main__":
    import uvicorn
    # Use the PORT environment variable provided by Render
    port = int(os.environ.get("PORT", 8000))
    # Bind to 0.0.0.0 to listen on all network interfaces
    uvicorn.run(app, host="0.0.0.0", port=port)
