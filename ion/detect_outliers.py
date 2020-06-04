#REMOVING OUTLIERS: removing data points not within 2-Standard deviations
def remove_outliers(df):
    temp = []
    for i in range(10): 
        print('Processing Batch-{}'.format(i+1))
        a = i * 500000
        b = (i+1) * 500000
        temp_df = df[a:b]
        plt.plot(temp_df['open_channels'],temp_df['signal'])
        plt.show()
        temp_df = temp_df[np.abs(temp_df.signal-temp_df.signal.mean()) <= (2*temp_df.signal.std())]
        print('Aftre removing outliers in Batch-{}'.format(i+1))
        plt.plot(temp_df['open_channels'],temp_df['signal'])
        plt.show()
        temp.append(temp_df)
        return temp
