# Osher Elhadad 318969748

def watch_data_info(data):

    # This function returns the first 5 rows for the object based on position.
    # It is useful for quickly testing if your object has the right type of data in it.
    print(data.head())

    # This method prints information about a DataFrame including the index dtype and column dtypes, non-null values and memory usage.
    print(data.info())

    # Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
    print(data.describe(include='all').transpose())


def print_data(data):
    print(f"number of users are :  {data['UserId'].nunique()}")
    print(f"number of products ranked are : {data['ProductId'].nunique()}")
    print(f"number of ranking are: {len(data['Rating'])}")
    print(f"minimum number of ratings given to a product : {data.groupby('ProductId')['Rating'].count().min()}")
    print(f"maximum number of ratings given to a product : {data.groupby('ProductId')['Rating'].count().max()}")
    print(f"minimum number of products ratings by user : {data.groupby('UserId')['Rating'].count().min()}")
    print(f"maximum number of products ratings by user : {data.groupby('UserId')['Rating'].count().max()}")



