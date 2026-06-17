from denario import Denario, models

den = Denario(project_dir="elm_test")
den.set_data_description("Toy test: analyze a small CSV of x,y points.")
den.get_idea(llm=models["gpt-4o"])
print(den.research.idea)
