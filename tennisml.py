#audio_plot_lib is used to generate audio graphs as an alternate option for screen reader users
import matplotlib.pyplot as plt
import audio_plot_lib as apl
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#loading the data and exploring
df = pd.read_csv("tennis_stats.csv")
df.columns = df.columns.str.lower()
print(df.info())
print(df.head(1))

#finding the best linear relationship feature
opportunity = np.array(df["breakpointsopportunities"])
winnings = np.array(df["winnings"])
#apl.interactive.plot(winnings, opportunity, script_name="tennisml.html")
plt.scatter(opportunity, winnings)
plt.title("break points opportunities vs winnings")
plt.xlabel("break-points-opportunities")
plt.ylabel("winnings")
plt.show()
plt.clf()

firstserve = np.array(df["firstserve"])
#apl.interactive.plot(winnings, firstserve, script_name="tennisml.html")
plt.scatter(firstserve, winnings)
plt.title("firstserve vs winnings")
plt.xlabel("firstserve")
plt.ylabel("winnings")
plt.show()
plt.clf()

firstservepointswon = np.array(df["firstservepointswon"])
#apl.interactive.plot(winnings, firstservepointswon, script_name="tennisml.html")
plt.scatter(firstservepointswon , winnings)
plt.title("firstservepointswon  vs winnings")
plt.xlabel("firstservepointswon ")
plt.ylabel("winnings")
plt.show()
plt.clf()

firstservereturnpointswon = np.array(df["firstservereturnpointswon"])
#apl.interactive.plot(winnings, firstservereturnpointswon, script_name="tennisml.html")
plt.scatter(firstservereturnpointswon , winnings)
plt.title("firstservereturnpointswon vs winnings")
plt.xlabel("firstservereturnpointswon")
plt.ylabel("winnings")
plt.show()
plt.clf()

secondservepointswon = np.array(df["secondservepointswon"])
#apl.interactive.plot(winnings,secondservepointswon , script_name="tennisml.html")
plt.scatter(secondservepointswon , winnings)
plt.title("secondservepointswon vs winnings")
plt.xlabel("secondservepointswon ")
plt.ylabel("winnings")
plt.show()
plt.clf()

aces = np.array(df["aces"])
#apl.interactive.plot(winnings, aces, script_name="tennisml.html")
plt.scatter(aces, winnings)
plt.title("aces vs winnings")
plt.xlabel("aces")
plt.ylabel("winnings")
plt.show()
plt.clf()

doublefaults = np.array(df["doublefaults"])
#apl.interactive.plot(winnings,doublefaults,title="double faults vs winnings", script_name="tennisml.html")
plt.scatter(doublefaults, winnings)
plt.title("doublefaults vs winnings")
plt.xlabel("doublefaults")
plt.ylabel("winnings")
plt.show()
plt.clf()

returngameswon = np.array(df["returngameswon"])
#apl.interactive.plot(winnings, returngameswon, title="return games won", script_name="tennisml.html")
plt.scatter(returngameswon, winnings)
plt.title("returngameswon vs winnings")
plt.xlabel("returngameswon")
plt.ylabel("winnings")
plt.show()
plt.clf()

servicegameswon = np.array(df["servicegameswon"])
#apl.interactive.plot(winnings, servicegameswon, script_name="tennisml.html")
plt.scatter(servicegameswon, winnings)
plt.title("servicegameswon vs winnings")
plt.xlabel("servicegameswon")
plt.ylabel("winnings")
plt.show()
plt.clf()

#linear regression for 1 feature
model = LinearRegression()
winnings = winnings.reshape(-1,1)
opportunity = opportunity.reshape(-1,1)
opportunity_train, opportunity_test, winnings_train, winnings_test= train_test_split(opportunity, winnings, train_size =0.8, test_size=0.2, random_state=6)
model.fit(opportunity_train, winnings_train)
predicted_opportunity_winnings= model.predict(opportunity_test)
print(model.coef_)
print("test score opportunity:",model.score(opportunity_test, winnings_test))
print("train score opportunity:",model.score(opportunity_train, winnings_train))
#apl.interactive.plot(x= winnings_test.reshape(-1), y= predicted_opportunity_winnings.reshape(-1), script_name="tennisml.html")
plt.scatter(winnings_test, predicted_opportunity_winnings)
plt.title("predictions vs actual winnings")
plt.xlabel("actual winnings")
plt.ylabel("predicted winnings")
plt.show()

model = LinearRegression()
aces = df[["aces"]]
aces_train, aces_test, winnings_train, winnings_test= train_test_split(aces, winnings, train_size =0.8, test_size=0.2, random_state=6)
model.fit(aces_train, winnings_train)
print("test score aces:",model.score(aces_test, winnings_test))
print("train score aces:",model.score(aces_train, winnings_train))

doublefaults = df[["doublefaults"]]
doublefaults_train, doublefaults_test, winnings_train, winnings_test= train_test_split(doublefaults, winnings, train_size =0.8, test_size=0.2, random_state=6)
model.fit(doublefaults_train, winnings_train)
print("test score doublefaults:",model.score(doublefaults_test, winnings_test))
print("train score doublefaults:",model.score(doublefaults_train, winnings_train))

#regression model with multi features, score_test without aces is same as with it
x= df[["aces","doublefaults", "breakpointsopportunities"]]
y = df[["winnings"]]
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, test_size=0.2, random_state=6)
lmodel = LinearRegression()
lmodel.fit(x_train, y_train)
prediction_y = lmodel.predict(x_test)
print(lmodel.coef_)
print("test with 3 features:",lmodel.score(x_test, y_test))
print("train with 3 features:",lmodel.score(x_train, y_train))

#regression with all independent features
x = df[["firstservepointswon", "firstservereturnpointswon", "secondservepointswon", "secondservereturnpointswon", "aces","breakpointsconverted", "breakpointsfaced", "breakpointsopportunities", "breakpointssaved", "doublefaults", "returnpointswon"]]
y = df[["winnings"]]
completemodel = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, test_size=0.2)
completemodel.fit(x_train, y_train)
print("test scorewith many features:", completemodel.score(x_test, y_test))
print("train score with many features:", completemodel.score(x_train, y_train))
predictionsofwinnings = completemodel.predict(x_test)
predictions = np.array(predictionsofwinnings).reshape(-1)
test_samples = np.array(y_test).reshape(-1)
apl.interactive.plot(predictions, test_samples, script_name="tennisml.html")
plt.scatter(y_test, predictionsofwinnings)
plt.title("predictions of winnings with several features")
plt.xlabel("actual")
plt.ylabel("predicted")
plt.show()

##breakpointsopportunities is having the strongest linear relationship, hence major impact on winnings