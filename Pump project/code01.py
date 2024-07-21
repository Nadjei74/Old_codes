import pandas as pd
# This code is used to check if a given waveform is viable for the voice coil and also calcules the duty cycle.
#input
df=pd.read_excel('Input.xlsx',sheet_name='Sheet1') # accepts current waveforms in an Excel format

df['S']=0.2308*(df['Current(A)']**2)+0.0385*df['Current(A)'] #this calculates the S value and stores it in the S column

# Calculating for time step value
df['change_t']=df['Time(t)'].diff().fillna(df['Time(t)']) #this clcautes duration for every cycle and stores them change_t column

#Calculating for the sum
last_val=df.loc[df.index[-1],'Time(t)'] #this picks the last time value from the time column
df['Cumm']=((df['change_t']*df['S']).cumsum())/last_val # this calculates the cummulative sum of S by Change in t_value

print(df)