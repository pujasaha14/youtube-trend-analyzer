{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "49f444b8",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# 1. Imports\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import re\n",
    "import joblib\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.metrics import classification_report, accuracy_score\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from scipy.sparse import hstack\n",
    "\n",
    "sns.set()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "e4ac3c2a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: joblib in c:\\users\\puja\\anaconda3\\lib\\site-packages (1.4.2)Note: you may need to restart the kernel to use updated packages.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "pip install joblib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "93afbc59",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>video_id</th>\n",
       "      <th>title</th>\n",
       "      <th>channel</th>\n",
       "      <th>category</th>\n",
       "      <th>tags</th>\n",
       "      <th>views</th>\n",
       "      <th>likes</th>\n",
       "      <th>comments</th>\n",
       "      <th>duration_sec</th>\n",
       "      <th>publish_date</th>\n",
       "      <th>engagement_ratio</th>\n",
       "      <th>success_level</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>vid_1</td>\n",
       "      <td>Top Education tutorial</td>\n",
       "      <td>channel_13</td>\n",
       "      <td>Education</td>\n",
       "      <td>live|trailer</td>\n",
       "      <td>97815</td>\n",
       "      <td>6252</td>\n",
       "      <td>624</td>\n",
       "      <td>558</td>\n",
       "      <td>2024-11-23</td>\n",
       "      <td>0.070295</td>\n",
       "      <td>High</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>vid_2</td>\n",
       "      <td>Guide Sports trailer</td>\n",
       "      <td>channel_280</td>\n",
       "      <td>Sports</td>\n",
       "      <td>official|vlog|challenge|trailer</td>\n",
       "      <td>89394</td>\n",
       "      <td>2099</td>\n",
       "      <td>786</td>\n",
       "      <td>529</td>\n",
       "      <td>2023-08-27</td>\n",
       "      <td>0.032272</td>\n",
       "      <td>Low</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>vid_3</td>\n",
       "      <td>Top News vlog</td>\n",
       "      <td>channel_14</td>\n",
       "      <td>News</td>\n",
       "      <td>reaction|challenge|tutorial|live</td>\n",
       "      <td>181069</td>\n",
       "      <td>3845</td>\n",
       "      <td>1761</td>\n",
       "      <td>830</td>\n",
       "      <td>2025-04-08</td>\n",
       "      <td>0.030960</td>\n",
       "      <td>Low</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>vid_4</td>\n",
       "      <td>Unbelievable Howto &amp; Style live</td>\n",
       "      <td>channel_389</td>\n",
       "      <td>Howto &amp; Style</td>\n",
       "      <td>tutorial|live|trailer</td>\n",
       "      <td>37465</td>\n",
       "      <td>1158</td>\n",
       "      <td>99</td>\n",
       "      <td>762</td>\n",
       "      <td>2025-02-20</td>\n",
       "      <td>0.033550</td>\n",
       "      <td>Low</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>vid_5</td>\n",
       "      <td>Best Music review</td>\n",
       "      <td>channel_48</td>\n",
       "      <td>Music</td>\n",
       "      <td>compilation|tutorial|review</td>\n",
       "      <td>358070</td>\n",
       "      <td>16441</td>\n",
       "      <td>1296</td>\n",
       "      <td>30</td>\n",
       "      <td>2023-02-14</td>\n",
       "      <td>0.049535</td>\n",
       "      <td>Medium</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  video_id                            title      channel       category  \\\n",
       "0    vid_1           Top Education tutorial   channel_13      Education   \n",
       "1    vid_2             Guide Sports trailer  channel_280         Sports   \n",
       "2    vid_3                    Top News vlog   channel_14           News   \n",
       "3    vid_4  Unbelievable Howto & Style live  channel_389  Howto & Style   \n",
       "4    vid_5                Best Music review   channel_48          Music   \n",
       "\n",
       "                               tags   views  likes  comments  duration_sec  \\\n",
       "0                      live|trailer   97815   6252       624           558   \n",
       "1   official|vlog|challenge|trailer   89394   2099       786           529   \n",
       "2  reaction|challenge|tutorial|live  181069   3845      1761           830   \n",
       "3             tutorial|live|trailer   37465   1158        99           762   \n",
       "4       compilation|tutorial|review  358070  16441      1296            30   \n",
       "\n",
       "  publish_date  engagement_ratio success_level  \n",
       "0   2024-11-23          0.070295          High  \n",
       "1   2023-08-27          0.032272           Low  \n",
       "2   2025-04-08          0.030960           Low  \n",
       "3   2025-02-20          0.033550           Low  \n",
       "4   2023-02-14          0.049535        Medium  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 2. Load Dataset\n",
    "# Use synthetic dataset (replace path with real API data after fetching)\n",
    "df = pd.read_csv(\"youtube_trends_full.csv\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "6ee83551",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 5000 entries, 0 to 4999\n",
      "Data columns (total 12 columns):\n",
      " #   Column            Non-Null Count  Dtype  \n",
      "---  ------            --------------  -----  \n",
      " 0   video_id          5000 non-null   object \n",
      " 1   title             5000 non-null   object \n",
      " 2   channel           5000 non-null   object \n",
      " 3   category          5000 non-null   object \n",
      " 4   tags              5000 non-null   object \n",
      " 5   views             5000 non-null   int64  \n",
      " 6   likes             5000 non-null   int64  \n",
      " 7   comments          5000 non-null   int64  \n",
      " 8   duration_sec      5000 non-null   int64  \n",
      " 9   publish_date      5000 non-null   object \n",
      " 10  engagement_ratio  5000 non-null   float64\n",
      " 11  success_level     5000 non-null   object \n",
      "dtypes: float64(1), int64(4), object(7)\n",
      "memory usage: 468.9+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3bc0eb87",
   "metadata": {},
   "source": [
    "# 3. Data Cleaning & Feature Engineering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "c5c27aa4",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.drop_duplicates(subset=['video_id'])\n",
    "for col in ['views','likes','comments']:\n",
    "    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "a7d0c338",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['title'] = df['title'].fillna('')\n",
    "df['tags'] = df['tags'].fillna('')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "e97c52c2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Engagement ratio\n",
    "df['engagement_ratio'] = (df['likes'] + df['comments']) / (df['views'] + 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "98aecb4e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Success label\n",
    "q1 = df['engagement_ratio'].quantile(0.33)\n",
    "q2 = df['engagement_ratio'].quantile(0.66)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "3bb4aa30",
   "metadata": {},
   "outputs": [],
   "source": [
    "def label(x):\n",
    "    if x >= q2:\n",
    "        return 'High'\n",
    "    elif x >= q1:\n",
    "        return 'Medium'\n",
    "    else:\n",
    "        return 'Low'\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "5b9aeb40",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>views</th>\n",
       "      <th>likes</th>\n",
       "      <th>comments</th>\n",
       "      <th>engagement_ratio</th>\n",
       "      <th>success_level</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>97815</td>\n",
       "      <td>6252</td>\n",
       "      <td>624</td>\n",
       "      <td>0.070295</td>\n",
       "      <td>High</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>89394</td>\n",
       "      <td>2099</td>\n",
       "      <td>786</td>\n",
       "      <td>0.032272</td>\n",
       "      <td>Low</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>181069</td>\n",
       "      <td>3845</td>\n",
       "      <td>1761</td>\n",
       "      <td>0.030960</td>\n",
       "      <td>Low</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>37465</td>\n",
       "      <td>1158</td>\n",
       "      <td>99</td>\n",
       "      <td>0.033550</td>\n",
       "      <td>Low</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>358070</td>\n",
       "      <td>16441</td>\n",
       "      <td>1296</td>\n",
       "      <td>0.049535</td>\n",
       "      <td>Medium</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    views  likes  comments  engagement_ratio success_level\n",
       "0   97815   6252       624          0.070295          High\n",
       "1   89394   2099       786          0.032272           Low\n",
       "2  181069   3845      1761          0.030960           Low\n",
       "3   37465   1158        99          0.033550           Low\n",
       "4  358070  16441      1296          0.049535        Medium"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['success_level'] = df['engagement_ratio'].apply(label)\n",
    "\n",
    "df[['views','likes','comments','engagement_ratio','success_level']].head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "77249df5",
   "metadata": {},
   "source": [
    "# 4. Exploratory Data Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "ead59f41",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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RERERERqYKAmIiIiIlIDAzURERERkRoYqImIiIiI1MBATURERESkBgZqIiIiIiI1MFATEREREamBgZqIiIiISA0M1EREREREamCgJiIiIiJSAwM1EREREZEaGKiJiIiIiNTAQE1EREREpAYGaiIiIiIiNTBQExERERGpgYGaiIiIiEgNDNRERERERGpgoCYiIiIiUgMDNRERERGRGhioiYiIiIjUwEBNRERERKQGBmoiIiIiIjUwUBMRERERqYGBmoiIiIhIDQzURERERERqYKAmIiIiIlIDAzURERERkRoYqImIiIiI1MBAnQtyuRxLliyBs7Mz6tevj379+iEmJkbssoiIiIhIRAzUubB8+XJs27YNfn5+CA0NhUQiwYABA5Ceni52aUREREQkEgbqHEpPT8fatWsxfPhwuLi4wNbWFgsXLsTLly9x5MgRscsjIiIiIpFIBEEQxC6iMLh+/Tq+//57HD58GBYWFortP/30E2xsbDB9+vRcjykIAuTyvLdfIgGkUikS36UhUybP8zjq0tWRonTJ4pDL5RD7u4k9UcWeqGJPVLEnyjSlHwB7kh32RBV7oio/eiKVSiCRSP77a+VteO0TGxsLAKhQoYLS9nLlyuHFixd5GlMikUBH57//kP5L6ZLF1R4jP0ilmvOGB3uiij1RxZ6oYk+UaUo/APYkO+yJKvZEVUH0RDO6XgikpKQAAPT09JS2Fy9eHGlpaWKUREREREQagIE6h/T19QFA5QbEtLQ0GBgYiFESEREREWkABuocyprqERcXp7Q9Li4O5ubmYpRERERERBqAgTqHbG1tYWRkhPPnzyu2JSUlISoqCg4ODiJWRkRERERi4k2JOaSnp4cePXogICAApqamqFSpEubPnw9zc3O4u7uLXR4RERERiYSBOhdGjBiBzMxMTJkyBampqXB0dERISIjKjYpEREREpD24DjURERERkRo4h5qIiIiISA0M1EREREREamCgJiIiIiJSAwM1EREREZEaGKiJiIiIiNTAQE1EREREpAYGaiIiIiIiNTBQExERERGpgU9KJCKiPJk7dy68vLxQs2ZNsUshKlTCwsKwZs0aPHz4EKGhodi5cyeqVq2Kzp07i12aKFJTUxEcHIx//vkHqamp+PczBzdu3ChSZTnHQE0EICgoCD4+PjAwMFDa/v79eyxevBiTJ08WqTLx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CwBYvXo1tm3bhgEDBqBTp04iV5ozfLALEREREWmMly9fQk9PDyYmJmKXkmMM1EREREREauBNiUREREREamCgJiIiIiJSAwM1EREREZEaGKiJiChbvMWGiChnGKiJiEhJUlISJkyYgMjISLFLISIqFBioiYhIya1bt7B7927I5XKxSyEiKhQYqImIiIiI1MBATURUxAiCgM2bN6NDhw6ws7ODu7s7goODFXOid+zYAS8vL9jb28POzg6dOnXCwYMHAQDnz59Hr169AAC9evVSPAoYAI4ePQovLy/Uq1cPzZs3h5+fHz58+KD0tU+ePAkvLy/Y2dmhbdu22L9/P9zd3bF06VLFMXFxcZg4cSJcXFxgZ2eHLl264NixY0rj2NjYICgoCN7e3mjUqBGCgoJQr149BAYGKh2XlpYGR0dHBAUF5V8DiYhyiY8eJyIqYgIDAxESEoI+ffqgefPmuHnzJhYuXIj09HSULl0afn5+GDZsGCZMmIDExEQEBwdj3LhxsLe3R506dTBt2jTMmDED06ZNQ5MmTQAA+/btw9ixY9GxY0eMHDkSz549w8KFC3H//n2sW7cOEokE586dw5AhQ+Dq6opffvkFMTEx+O2335CWlqao7fXr1+jSpQuKFSuGUaNGwcTEBLt27cLQoUMxb948fPfdd4pjV6xYgV9++QU2NjYwNzdHdHQ09u3bh1GjRkEikQAAjh07hnfv3qFz584F2mMiok8xUBMRFSFJSUlYt24devbsifHjxwMAmjdvjoSEBFy6dAnW1tbo168fhg4dqjincuXK8PLywuXLl+Hh4QErKysAgJWVFaysrCAIAgICAuDs7IyAgADFedWrV0efPn0QFhaGVq1aYenSpbCyskJQUJAi8JqZmWH06NGKc9atW4eEhAQcOnQIVapUAQC4uLigT58+mDdvHjw8PCCVfnzz1M7ODj///LPiXG9vbxw8eBDnz5+Hk5MTAODPP/9EkyZNULly5a/RTiKiHGGgJiIqQq5evYqMjAy4u7srbf/111+VPn/37h0ePXqER48eISIiAgCQkZGR7ZgPHjxAbGwsBg4ciMzMTMV2R0dHGBkZITw8HM2aNcOVK1cwdOhQRZgGgLZt20JX93//1Fy4cAENGjRQhOks3333HSZOnIgHDx4oAr21tbXSMc2aNUPFihWxZ88eODk5IS4uDuHh4Zg1a1ZO20NE9FVwDjURURGSmJgIADA1Nc12/+PHj9GnTx84Ojrip59+QnBwsCJIf27d6awxfX19UadOHaWP9+/fIy4uDomJiZDJZDAzM1M6V1dXFyYmJorP3759izJlyqh8jaxtSUlJKtuySKVSeHl54a+//kJqair27t0LfX19tG3b9gsdISL6+niFmoioCDE2NgYAJCQkoEaNGortL168wKNHjzB16lQYGBhg+/btqF27NnR1dXH//n3s3bv3P8ccP348GjdurLK/VKlSMDMzQ7FixRAfH6+0Ty6X482bN0rHvn79WmWMV69eAYBS+M6Ol5cXli1bhlOnTuHgwYNo3/7/2rl/kGTXMI7j34Nkf6C/L4Jg9EcihcjBHApqiLZDEIFTYOAQNZhlZTW15ebgoktQklFjSLhkEWHQ1mBF4NCghDVG0BDUGQ4IEafDy1O857z8PvBsz3093Pf04+Z6rj+pra39dI2IyHfTDbWIyG/E5XJRVVX1YWpGMpnE7/dTLBbxer24XK5KK8bp6SlAZe60yWR6t9Zut/Pjxw9KpRK9vb2Vx2q1Eo1Gub6+xmQy4Xa7yWaz79YeHx9/aBO5uLigWCy+ey+dTmOxWGhvb/90fzabjYGBAba3t7m6umJ8fPwnTkdE5HvohlpE5DfS0tLC5OQkyWQSs9lMf38/+XyeVCrF8vIyqVSKnZ0drFYrDQ0N5HI5kskkAM/PzwDU19cDf4/Aa2xsxOl0EgqFWFtbw2QyMTw8zOPjI/F4nPv7e3p6egAIBoP4fD6CwSBer5e7uztisRhApa/a7/eTTqfx+/0EAgGam5vZ39/n/PycSCRS+SHxM16vl4WFBTo6Oujr6/vyMxQR+Vl/vP1T05yIiPwvvb29sbm5ye7uLuVymdbWVnw+HxMTE9zc3LC+vs7l5SVms5muri5mZmaIRCJ0d3cTi8V4fX0lHA5zeHhIW1sbBwcHAGQyGTY2NigUCtTV1eF2u5mfn8fhcFS+nc1micVi3N7eYrPZmJubIxQKsbq6it/vB6BYLBKNRjk7O+Pl5QWn08nU1BQjIyOVOg6Hg0AgwOzs7If9PT094fF4CIVCTE9Pf/Npioj8OwVqERH5EkdHR1it1sqNNUChUGB0dJR4PP4uMBuRyWQIh8OcnJxgsVi+pKaIiBFq+RARkS+Ry+XIZDIsLS3R2dlJuVwmkUhgt9sZHBw0XD+bzZLP59nb22NsbExhWkT+MxSoRUTkS6ysrFBTU0MikeDh4YGmpiaGhoZYXFykurracP1SqcTW1hYej+fDXG0RkV9JLR8iIiIiIgZobJ6IiIiIiAEK1CIiIiIiBihQi4iIiIgYoEAtIiIiImKAArWIiIiIiAEK1CIiIiIiBihQi4iIiIgYoEAtIiIiImLAX0vK1ouJUoaLAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 800x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Average views by category\n",
    "by_cat = df.groupby('category')['views'].mean().sort_values(ascending=False)\n",
    "plt.figure(figsize=(8,5))\n",
    "by_cat.plot.bar()\n",
    "plt.title('Average Views by Category')\n",
    "plt.ylabel('Average Views')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "0d2b673f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAeIAAAH2CAYAAACsro8uAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8qNh9FAAAACXBIWXMAAA9hAAAPYQGoP6dpAABoe0lEQVR4nO3dd3wUZeIG8GdmS3bTeyGFEAiB0HtvEVAQkOaBYgHFfidi4zz7nXf+PMsdJyJKExUsCILSFZBepHcIhJaQ3nt2Z+b3R2A1EiCEJO+W5/vRj7L12c2yT+add96RNE3TQERERELIogMQERG5MhYxERGRQCxiIiIigVjEREREArGIiYiIBGIRExERCcQiJiIiEohFTEREJBCLmIjqlT2sGWQPGYiuhUVMt+zUqVOYOnUqevXqhdatW6N379545plncOzYMdHRGtTSpUsRFxeH5ORk0VEA1CxPQkIC4uLibP+2bNkSnTt3xj333IPly5dfdfu4uDh8+OGHNc6wePFivPPOOze83f3334/777+/1s9zLQUFBZg2bRr27NlzzeciEk0vOgA5tsTERIwbNw5t27bFyy+/jMDAQKSlpeHLL7/EuHHj8MUXX6B9+/aiY9J19OvXD08++SQAwGq1Ijc3F6tWrcKLL76IEydOYNq0abbbfvPNNwgNDa3xY3/88cfo2rXrDW/3+uuv33zwGjh+/DiWLVuG0aNH1/tzEdUWi5huyfz58+Hr64s5c+bAYDDYLh84cCCGDBmCmTNn4tNPPxWYkG7E39//ql+WBg0ahICAAMybNw8DBw5Ep06dAKDefqlq1qxZvTyu6OciqgkOTdMtycrKAnD1Pjh3d3e89NJLGDJkiO2y6oYEd+3ahbi4OOzatct22YULF/D000+ja9eu6NKlCx555BEkJibari8uLsbbb7+Nvn37on379hg9ejQ2bNhQ5XEXL16MO++8E61bt0b//v3x4Ycfwmq12q7PycnB888/j169eqFNmza46667sGzZMtv1qqpi+vTpSEhIQOvWrZGQkIAPPvgAFoul9m/WZXl5eXjttdfQs2dPtGnTBn/605+wY8cO2/UPPfQQRo4cedX9nnnmGdx55522P+/Zswf33Xcf2rVrh65du2LatGnIycm55XxXPP300zAajfj6669tl/1xyPiLL77AHXfcgTZt2qBPnz544403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YBHTLenfvz++/fZbzJ07F7NmzUJOTg6MRiPi4+Pxn//8B4MHD7bddsyYMbhw4QK+//57fPPNN+jatSumT5+Oe+65x3absLAwLFq0CO+++y5eeuklGI1GdO3aFe+++y58fX0BVB5a9P777+PDDz9ESUkJWrRogTlz5qBDhw4AKrfUgoKCsGjRIsyZMwc+Pj7o0aMHnn32WdsX84wZM/DBBx9g+vTpyM3NRVhYGP785z/b9i1PmTIFRqMRS5YswUcffQQvLy8kJCTgueeeu+F7MnPmzGovnzhxItzd3bFw4UK8//77ePfdd1FYWGjbr/rQQw9VuX3z5s3RsmVLpKeno1evXlWu6927N+bOnYsZM2bg6aefhsFgQKtWrTB//vybXnRj06ZNthEGvV6PwMBANG/eHLNmzaoyKe6Pxo8fD4vFgq+//hqLFi2CyWRCjx498MILL9h+oXjooYfwr3/9Cw8//DDmz59/U7meeOIJHDt2DI8//ji8vLwwefJk2xY4ULPPU2xsLIYNG4aFCxdiy5YtWLFiRZXnkCQJn3zyCf73v//h888/R05ODiIiIjB16lRMmjTppvIS1ZakcTV0IiIiYbiPmIiISCAWMRERkUAsYiIiIoFYxERERAKxiImIiARiERMREQnEIiYiIhKIRUxERCQQi5iIiEggFjEREZFALGIiIiKBWMREREQCsYiJiIgEYhETEREJxCImIiISiEVMREQkEIuYiIhIIBYxERGRQCxiIiIigVjEREREArGIiYiIBGIRExERCcQiJiIiEohFTEREJBCLmIiISCAWMRERkUAsYiIiIoFYxERERAKxiImIiARiERMREQnEIiYiIhKIRUxERCQQi5iIiEggFjEREZFALGIiIiKBWMREREQCsYiJiIgEYhETEREJxCImIiISiEVMREQkEIuYiIhIIBYxERGRQCxiIiIigVjEREREArGIiYiIBGIRExERCcQiJiIiEohFTEREJBCLmIiISCAWMRERkUAsYiIiIoFYxERERAKxiImIiARiERMJpGkarIoKq6JC07Tr3lZRVVRYFJSWW1FUWoH8onLkFJQhK68U+UXlKCmzwGJVa/zcVkWFxapCvcHzUs3df//9+Otf/1rtdX/9619x//33AwDi4uKwdOnSGj1mcnIy4uLisGvXrjrLSfZFLzoAkbNSVQ2qpkEnS5Ak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uj6jzoAYv8ZoFtAYsf5NEBfYFP7uvgBQZauXyNWxiImuQVFVSJeLucxajov5l5CUewEX8lJwPi8FF/JTUMbzKQOo3C8f7hWK2IBoNAtogriAGET4hEGWZKiaCk3THHqCFVF9YhET3QSraoVO0tkO6cgqzkFS7gWcz0vG+bwUnM9PQUZRltMOa+tlPUI8A9HIKwShnsFo5BWMcO9QRPtFwqR3g6ZpUDQFepl7vYhqikVMdItUVYWG37b4KqwVSC3KQEZRNrJKcpBVkovsklxkleQguyQXuWX5tv2i9kgnyQh090eYVwjCvIIR5hWMRl4hCPcOhb/Z1/ZLiKIq0OAa+3SJ6hOLmKieqJoGVVUqD6f6XVmpmor8skJkleQgoygLWSW5KCgvQpm1DKXWMpRZylFqLUeZ7f/LUGat/O/N/HWVJRkGWQ+9Tg93gxnebp7wdvOE1+X/ert52f7fz+QDb5MXvIwetvP0XsmqamqVUQAiqlssYiKBFFWFhsqFKmRJvmHZWRQLypWKyv3XACBJkCGh8p/Kx9DLeuhk+brH3V55XkCCrgbPS0T1h0VMREQkkHMsVUNEROSgWMREREQCsYiJiIgEYhETEREJxCImIiISiEVMREQkEIuYiIhIIBYxERGRQCxiIiIigVjEREREArGIiYiIBGIRExERCcQiJiIiEohFTEREJBCLmIiISCAWMRERkUAsYiIiIoFYxERERAKxiImIiARiERMREQnEIiYiIhKIRUxERCQQi5iIiEggFjEREZFALGIiIiKBWMREREQCsYiJiIgEYhETEREJxCImIiISiEVMREQkEIuYiIhIIBYxERGRQCxiIiIigVjEREREArGIiYiIBGIRExERCcQiJiIiEohFTEREJBCLmIiISCAWMRERkUAsYiIiIoFYxERERAKxiImIiARiERMREQnEIiYiIhKIRUxERCQQi5iIiEggFjEREZFALGIiIiKBWMREREQCsYiJiIgEYhETEREJxCImIiISiEVMREQkEIuYiIhIoP8HzPGYAl1XMkkAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 600x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Success level distribution\n",
    "plt.figure(figsize=(6,6))\n",
    "df['success_level'].value_counts().plot.pie(autopct='%1.1f%%')\n",
    "plt.title('Success Level Distribution')\n",
    "plt.ylabel('')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "6efa7e18",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 600x400 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Correlation heatmap\n",
    "plt.figure(figsize=(6,4))\n",
    "sns.heatmap(df[['views','likes','comments','engagement_ratio']].corr(), annot=True, cmap='coolwarm')\n",
    "plt.title(\"Feature Correlation\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "08632d96",
   "metadata": {},
   "source": [
    "# 5. Machine Learning — RandomForest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "73a4dd6e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Encode categorical\n",
    "df['text'] = df['title'] + ' ' + df['tags']\n",
    "le = LabelEncoder()\n",
    "y = le.fit_transform(df['success_level'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "47033230",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Text vectorization\n",
    "tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))\n",
    "X_text = tfidf.fit_transform(df['text'])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "6a0f30b2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Numeric features\n",
    "numeric = df[['views','likes','comments']].fillna(0)\n",
    "X = hstack([X_text, numeric.values])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "91061d9e",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "61a5a67e",
   "metadata": {},
   "outputs": [],
   "source": [
    "clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)\n",
    "clf.fit(X_train, y_train)\n",
    "y_pred = clf.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "afa0d72d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.681\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "        High       0.69      0.74      0.72       340\n",
      "         Low       0.75      0.80      0.77       330\n",
      "      Medium       0.58      0.50      0.54       330\n",
      "\n",
      "    accuracy                           0.68      1000\n",
      "   macro avg       0.67      0.68      0.68      1000\n",
      "weighted avg       0.67      0.68      0.68      1000\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred))\n",
    "print(classification_report(y_test, y_pred, target_names=le.classes_))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "de324cd0",
   "metadata": {},
   "source": [
    "# 6. Save Model\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "4cbadd29",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "58ef8cb8",
   "metadata": {},
   "outputs": [],
   "source": [
    "os.makedirs(\"models\", exist_ok=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "7e5fa562",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model saved to models/rf_success.pkl\n"
     ]
    }
   ],
   "source": [
    "joblib.dump({\n",
    "    'model': clf,\n",
    "    'tfidf': tfidf,\n",
    "    'label_encoder': le\n",
    "}, 'models/rf_success.pkl')\n",
    "\n",
    "print(\"Model saved to models/rf_success.pkl\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4545d90d",
   "metadata": {},
   "source": [
    "# 7. Insights"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "0ea290dc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Top categories by average views:\n",
      " category\n",
      "Music            329914.483221\n",
      "Entertainment    222873.538071\n",
      "Gaming           128953.577371\n",
      "Sports           123593.421801\n",
      "Comedy            97085.072727\n",
      "Name: views, dtype: float64\n"
     ]
    }
   ],
   "source": [
    "# Top performing categories\n",
    "print(\"Top categories by average views:\\n\", by_cat.head(5))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "19003121",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "success_level\n",
      "High      0.075473\n",
      "Low       0.035248\n",
      "Medium    0.055102\n",
      "Name: engagement_ratio, dtype: float64\n"
     ]
    }
   ],
   "source": [
    "# Engagement distribution summary\n",
    "print(df.groupby('success_level')['engagement_ratio'].mean())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "0a5e7a45",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Prediction for sample video: ['Medium']\n"
     ]
    }
   ],
   "source": [
    "# Example prediction pipeline\n",
    "sample = [\"Amazing Music Official Trailer\"]\n",
    "X_sample = tfidf.transform(sample)\n",
    "X_num = np.array([[100000, 5000, 200]])  # views, likes, comments\n",
    "X_input = hstack([X_sample, X_num])\n",
    "pred = clf.predict(X_input)\n",
    "print(\"Prediction for sample video:\", le.inverse_transform(pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "c7d0b0bc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting streamlitNote: you may need to restart the kernel to use updated packages.\n",
      "\n",
      "  Downloading streamlit-1.49.1-py3-none-any.whl (10.0 MB)\n",
      "     ---------------------------------------- 10.0/10.0 MB 3.2 MB/s eta 0:00:00\n",
      "Requirement already satisfied: click<9,>=7.0 in c:\\users\\puja\\anaconda3\\lib\\site-packages (from streamlit) (8.0.4)\n",
      "Collecting protobuf<7,>=3.20\n",
      "  Downloading protobuf-6.32.0-cp39-cp39-win_amd64.whl (435 kB)\n",
      "     -------------------------------------- 435.8/435.8 kB 4.5 MB/s eta 0:00:00\n",
      "Requirement already satisfied: numpy<3,>=1.23 in c:\\users\\puja\\anaconda3\\lib\\site-packages (from streamlit) (1.24.4)\n",
      "Collecting altair!=5.4.0,!=5.4.1,<6,>=4.0\n",
      "  Downloading altair-5.5.0-py3-none-any.whl (731 kB)\n",
      "     -------------------------------------- 731.2/731.2 kB 3.3 MB/s eta 0:00:00\n",
      "Collecting blinker<2,>=1.5.0\n",
      "  Downloading blinker-1.9.0-py3-none-any.whl (8.5 kB)\n",
      "Requirement already satisfied: toml<2,>=0.10.1 in c:\\users\\puja\\anaconda3\\lib\\site-packages (from streamlit) (0.10.2)\n",
      "Collecting gitpython!=3.1.19,<4,>=3.0.7\n",
      "  Downloading gitpython-3.1.45-py3-none-any.whl (208 kB)\n",
      "     ------------------------------------ 208.2/208.2 kB 975.0 kB/s eta 0:00:00\n",
      "Collecting pydeck<1,>=0.8.0b4\n",
      "  Downloading pydeck-0.9.1-py2.py3-none-any.whl (6.9 MB)\n",
      "     ---------------------------------------- 6.9/6.9 MB 3.3 MB/s eta 0:00:00\n",
      "Requirement already satisfied: packaging<26,>=20 in c:\\users\\puja\\anaconda3\\lib\\site-packages (from streamlit) (21.3)\n",
      "Requirement already satisfied: requests<3,>=2.27 in c:\\users\\puja\\anaconda3\\lib\\site-packages (from streamlit) (2.32.3)\n",
      "Requirement already satisfied: pillow<12,>=7.1.0 in c:\\users\\puja\\anaconda3\\lib\\site-packages (from streamlit) (9.2.0)\n",
      "Requirement already satisfied: cachetools<7,>=4.0 in c:\\users\\puja\\anaconda3\\lib\\site-packages (from streamlit) (5.5.0)\n",
      "Collecting tenacity<10,>=8.1.0\n",
      "  Downloading tenacity-9.1.2-py3-none-any.whl (28 kB)\n",
      "Requirement already satisfied: pyarrow>=7.0 in c:\\users\\puja\\anaconda3\\lib\\site-packages (from streamlit) (18.1.0)\n",
      "Requirement already satisfied: typing-extensions<5,>=4.4.0 in c:\\users\\puja\\anaconda3\\lib\\site-packages (from streamlit) (4.12.2)\n",
      "Requirement already satisfied: watchdog<7,>=2.1.5 in c:\\users\\puja\\anaconda3\\lib\\site-packages (from streamlit) (2.1.6)\n",
      "Requirement already satisfied: tornado!=6.5.0,<7,>=6.0.3 in c:\\users\\puja\\anaconda3\\lib\\site-packages (from streamlit) (6.1)\n",
      "Requirement already satisfied: pandas<3,>=1.4.0 in c:\\users\\puja\\anaconda3\\lib\\site-packages (from streamlit) (1.4.4)\n",
      "Requirement already satisfied: jinja2 in c:\\users\\puja\\anaconda3\\lib\\site-packages (from altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit) (2.11.3)\n",
      "Collecting narwhals>=1.14.2\n",
      "  Downloading narwhals-2.3.0-py3-none-any.whl (404 kB)\n",
      "     ------------------------------------ 404.4/404.4 kB 721.8 kB/s eta 0:00:00\n",
      "Requirement already satisfied: jsonschema>=3.0 in c:\\users\\puja\\anaconda3\\lib\\site-packages (from altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit) (4.16.0)\n",
      "Requirement already satisfied: colorama in c:\\users\\puja\\anaconda3\\lib\\site-packages (from click<9,>=7.0->streamlit) (0.4.5)\n",
      "Collecting gitdb<5,>=4.0.1\n",
      "  Downloading gitdb-4.0.12-py3-none-any.whl (62 kB)\n",
      "     ---------------------------------------- 62.8/62.8 kB 1.7 MB/s eta 0:00:00\n",
      "Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in c:\\users\\puja\\anaconda3\\lib\\site-packages (from packaging<26,>=20->streamlit) (3.0.9)\n",
      "Requirement already satisfied: python-dateutil>=2.8.1 in c:\\users\\puja\\anaconda3\\lib\\site-packages (from pandas<3,>=1.4.0->streamlit) (2.8.2)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\users\\puja\\anaconda3\\lib\\site-packages (from pandas<3,>=1.4.0->streamlit) (2022.1)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\users\\puja\\anaconda3\\lib\\site-packages (from requests<3,>=2.27->streamlit) (2.0.4)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\puja\\anaconda3\\lib\\site-packages (from requests<3,>=2.27->streamlit) (3.3)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\users\\puja\\anaconda3\\lib\\site-packages (from requests<3,>=2.27->streamlit) (1.26.11)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\puja\\anaconda3\\lib\\site-packages (from requests<3,>=2.27->streamlit) (2022.9.14)\n",
      "Collecting smmap<6,>=3.0.1\n",
      "  Downloading smmap-5.0.2-py3-none-any.whl (24 kB)\n",
      "Requirement already satisfied: MarkupSafe>=0.23 in c:\\users\\puja\\anaconda3\\lib\\site-packages (from jinja2->altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit) (2.0.1)\n",
      "Requirement already satisfied: attrs>=17.4.0 in c:\\users\\puja\\anaconda3\\lib\\site-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit) (21.4.0)\n",
      "Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in c:\\users\\puja\\anaconda3\\lib\\site-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit) (0.18.0)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\puja\\anaconda3\\lib\\site-packages (from python-dateutil>=2.8.1->pandas<3,>=1.4.0->streamlit) (1.16.0)\n",
      "Installing collected packages: tenacity, smmap, protobuf, narwhals, blinker, pydeck, gitdb, altair, gitpython, streamlit\n",
      "  Attempting uninstall: tenacity\n",
      "    Found existing installation: tenacity 8.0.1\n",
      "    Uninstalling tenacity-8.0.1:\n",
      "      Successfully uninstalled tenacity-8.0.1\n",
      "Successfully installed altair-5.5.0 blinker-1.9.0 gitdb-4.0.12 gitpython-3.1.45 narwhals-2.3.0 protobuf-6.32.0 pydeck-0.9.1 smmap-5.0.2 streamlit-1.49.1 tenacity-9.1.2\n"
     ]
    }
   ],
   "source": [
    "pip install streamlit"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "24e00029",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import joblib\n",
    "import numpy as np\n",
    "from scipy.sparse import hstack\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b60fe1ec",
   "metadata": {},
   "source": [
    "# Load model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "7188a833",
   "metadata": {},
   "outputs": [],
   "source": [
    "artifacts = joblib.load(\"models/rf_success.pkl\")\n",
    "clf = artifacts['model']\n",
    "tfidf = artifacts['tfidf']\n",
    "le = artifacts['label_encoder']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "5fd9d5c9",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-09-06 16:20:43.927 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:44.707 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\PUJA\\anaconda3\\lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-09-06 16:20:44.716 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:44.723 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "st.title(\"🎬 YouTube Trend Analyzer\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "5e56a786",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-09-06 16:20:47.888 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:47.891 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:47.892 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:47.901 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:47.902 Session state does not function when running a script without `streamlit run`\n",
      "2025-09-06 16:20:47.905 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:47.907 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:47.910 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:47.914 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:47.917 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:47.921 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:47.924 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:47.926 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:47.928 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:47.932 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:47.934 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:47.941 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:47.945 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:47.950 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:47.953 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:47.955 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:47.958 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:47.961 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:47.965 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:47.968 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:47.971 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:47.973 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:47.975 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:47.982 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "title = st.text_input(\"Video Title\", \"Amazing Music Official Trailer\")\n",
    "views = st.number_input(\"Views\", 0, 10000000, 50000)\n",
    "likes = st.number_input(\"Likes\", 0, 1000000, 3000)\n",
    "comments = st.number_input(\"Comments\", 0, 100000, 200)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "41546771",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-09-06 16:20:53.035 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:53.038 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:53.040 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:53.043 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:53.049 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-06 16:20:53.051 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "if st.button(\"Predict Success Level\"):\n",
    "    X_text = tfidf.transform([title])\n",
    "    X_num = np.array([[views, likes, comments]])\n",
    "    X = hstack([X_text, X_num])\n",
    "    pred = clf.predict(X)\n",
    "    st.success(f\"Predicted Success Level: {le.inverse_transform(pred)[0]}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5fea376a",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
