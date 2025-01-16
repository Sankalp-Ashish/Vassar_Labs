import streamlit as st
import faiss
import numpy as np
import openai
import os
import requests
from sentence_transformers import SentenceTransformer
 
# Initialize OpenAI API key
os.environ["OPENAI_API_KEY"] = "REMOVED_OPENAI_KEY"  # Replace with your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
 
# Initialize SentenceTransformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
 
# Initialize FAISS index for storing recipe embeddings
embedding_dim = 384  # Based on the SentenceTransformer model
index = faiss.IndexFlatL2(embedding_dim)
 
# Recipe data
recipes = [
    {"name": "Chocolate Cake", "description": "A rich chocolate dessert."},
    {"name": "Vanilla Ice Cream", "description": "A creamy vanilla treat."},
    {"name": "Apple Pie", "description": "A classic dessert with apples."},
    {"name": "Vegetable Stir Fry", "description": "A healthy stir fry with mixed vegetables."},
    {"name": "Chicken Curry", "description": "A flavorful spicy chicken curry."},
    {"name": "Caesar Salad", "description": "A fresh salad with romaine lettuce and Caesar dressing."},
    {"name": "Gluten-Free Pancakes", "description": "Fluffy pancakes made without gluten."},
    {"name": "Vegetarian Tacos", "description": "Tacos filled with seasoned vegetables and beans."},
    {"name": "Spaghetti Carbonara", "description": "A creamy pasta dish with eggs, cheese, and pancetta."},
    {"name": "Beef Stew", "description": "A hearty stew made with tender beef and vegetables."}
]
 
# Add recipes to FAISS
recipe_embeddings = [model.encode(recipe["description"]) for recipe in recipes]
recipe_embeddings_np = np.array(recipe_embeddings).astype('float32')
index.add(recipe_embeddings_np)
 
# Function to query FAISS for recipe suggestions
def query_faiss(query, k=3):
    query_embedding = model.encode(query).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    return distances, [recipes[i] for i in indices[0]]
 
# Function to fetch recipes from Spoonacular API
def fetch_recipes_from_api(query):
    API_KEY = "REMOVED_SPOONACULAR_API_KEY"  # Replace with your Spoonacular API key
    url = f"https://api.spoonacular.com/recipes/complexSearch"
    params = {"query": query, "number": 10, "apiKey": API_KEY}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get("results", [])
    else:
        return []
 
# Function to generate detailed recipe with OpenAI
def generate_recipe_with_openai(recipe_name, ingredients, steps):
    prompt = (
        f"Generate a detailed recipe for {recipe_name}.\n\n"
        f"Ingredients:\n{', '.join(ingredients)}\n\n"
        f"Steps:\n{steps}\n"
    )
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful recipe assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )
    return response['choices'][0]['message']['content']
 
# Enhanced Intent Classification Function
def classify_intent(user_input):
    user_input = user_input.lower()
    keywords = {
        "recipe_request": ["recipe", "cook", "bake", "prepare", "dish", "food"],
        "ingredient_query": ["ingredient", "what do I need", "what's in", "needed for"],
        "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
    }
 
    for intent, keyword_list in keywords.items():
        if any(keyword in user_input for keyword in keyword_list):
            return intent
 
    return "unknown"
 
# Streamlit UI
st.title("Recipe Chat System with Intent Classification")
 
user_input = st.text_input("What would you like to ask?")
 
if user_input:
    # Intent Classification
    intent = classify_intent(user_input)
    st.write(f"Intent: {intent}")
 
    if intent == "recipe_request":
        st.subheader("Recipe Suggestions from FAISS:")
        distances, top_recipes = query_faiss(user_input)
 
        # Display recipes with their names, descriptions, and scores
        for i, recipe in enumerate(top_recipes):
            distance = distances[0][i]  # Get the distance for the i-th recipe
            score = 1 / (1 + distance)  # Inverse of distance (higher score is better)
            st.write(f"**{recipe['name']}** (Score: {score:.2f})")
            st.write(f"Description: {recipe['description']}")
 
        # Fetch additional recipes from the Spoonacular API
        st.subheader("Additional Recipes from Spoonacular API:")
        api_recipes = fetch_recipes_from_api(user_input)
        for recipe in api_recipes:
            st.write(f"**{recipe['title']}**")
            st.write(f"Description: {recipe.get('summary', 'No description available.')}")
        
        # Example OpenAI prompt for generating a detailed recipe
        # if top_recipes:
        #     st.subheader("Generated Recipe (from OpenAI):")
        #     example_recipe = top_recipes[0]  # Choose the first recommended recipe
        #     ingredients = ["flour", "milk", "eggs", "sugar", "baking powder"]  # Example ingredients
        #     steps = "1. Preheat your oven to 350°F (175°C). 2. Mix ingredients. 3. Bake for 25-30 minutes."  # Example steps
        #     detailed_recipe = generate_recipe_with_openai(
        #         example_recipe["name"], ingredients, steps
        #     )
        #     st.write(detailed_recipe)
 
    if intent == "ingredient_query":
        st.subheader("Ingredient Information")
        st.write("You asked about ingredients. Please specify the dish or recipe.")
 
    elif intent == "greeting":
        st.subheader("Greetings!")
        st.write("Hello! How can I assist you today?")
 
    else:
        st.write("Sorry, I didn't understand that. Can you try rephrasing?")
        #recipe and intent classification