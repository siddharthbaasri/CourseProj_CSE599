import pandas as pd
import ast

def get_action(actions_str):
    actions = ast.literal_eval(actions_str)
    return actions[0]['action']

def sample_df(df, approx_size = 1000, random_seed = 42):
    #To avoid complications in our analysis, we choose to only keep datapoints with one main action.
    df = df[df['numerosity'] == 1].reset_index(drop=True)

    #Extract the action for each brow of the dataframe.
    df['main_action'] = df["actions"].apply(get_action)
    #We use stratified random sampling. This is done to ensure that action type proportions remain the same
    sampled_df = df.groupby('main_action', group_keys=False).apply( 
        lambda x: x.sample(n=min(len(x), int(approx_size * len(x) / len(df))), random_state=random_seed) 
    )
    sampled_df['main_action'] = sampled_df['actions'].apply(get_action)

    #Note that it can return a value slightly less due to rounding
    print(f"Total samples: {len(sampled_df)}")
    print(df['main_action'].value_counts())
    print(sampled_df['main_action'].value_counts()) 

    return sampled_df

def main():
    #The data donwloaded from the git hub repo.
    original_data_path = './data/dataset.csv'
    sampled_file_path = './data/sampled_data.csv'

    df = pd.read_csv(original_data_path, sep=';')

    #Sample the 
    sample_size = 1000
    sampled_df = sample_df(df, sample_size)

    #Save the new df to the file path. This sampled dataset is being used.
    sampled_df.to_csv(sampled_file_path, sep=';', index=False)


if __name__ == "__main__":
    main()