def create_result_folders(base_dir='results'):
    from datetime import datetime
    import os
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_folder = f'{base_dir}/{timestamp}'
    os.makedirs(f'{result_folder}/weights', exist_ok=True)
    os.makedirs(f'{result_folder}/plots', exist_ok=True)
    return result_folder
