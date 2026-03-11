#ausreddit exploration v2

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pyarrow.parquet as pq
import psutil
import os
import ipywidgets as widgets
from IPython.display import display


class ExplorationAR:
    def __init__(self) -> None:
        pass

    # ngrams

    def top_ngrams_in_date_range(self, file_path: str, n: int, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get top n n-grams in a specified time frame by summing their counts.
        Reads data from a Parquet file in a memory-efficient way.
        Assumes data has a 'month' column (YYYY-MM-01 format), 
        an 'ngram' column (containing single or multi-word n-grams as strings),
        and a 'count' column (frequency of that ngram for the month).
        """
        try:
            # Convert start_date and end_date strings to datetime objects for comparison
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            # Open the Parquet file for streaming
            table = pq.ParquetFile(file_path)

            # Initialize an empty DataFrame to collect filtered results
            filtered_data = []

            # Stream through the Parquet file in row groups
            for batch in table.iter_batches(columns=['month', 'ngram', 'count']):
                df = batch.to_pandas()

                # Ensure 'month' column is in datetime format
                df['month'] = pd.to_datetime(df['month'])

                # Filter rows based on the date range
                mask = (df['month'] >= start_dt) & (df['month'] <= end_dt)
                filtered_data.append(df[mask])

            # Concatenate all filtered chunks
            filtered_data = pd.concat(filtered_data, ignore_index=True)

            if filtered_data.empty:
                print(f"No data found in the date range {start_date} to {end_date}.")
                return pd.DataFrame(columns=['ngram', 'total_count'])

            # Group by 'ngram' and sum the 'count' for the filtered period
            ngram_counts = filtered_data.groupby('ngram')['count'].sum()

            # Get the top N n-grams
            top_n_ngrams = ngram_counts.nlargest(n).reset_index(name='total_count')

            if not top_n_ngrams.empty:
                print(self._get_ram_status())

            return top_n_ngrams
        except KeyError as e:
            print(f"Error processing n-grams: Missing expected column {e}. Ensure 'month', 'ngram', and 'count' columns exist.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error processing n-grams: {e}")
            return pd.DataFrame()

    def keyword_search_in_date_range(self, file_path: str, keyword: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Search for a keyword (as a whole word, case-insensitive) within the 'ngram' column 
        in a specified time frame.
        Reads data from a Parquet file in a memory-efficient way.
        Assumes data has a 'month' column (YYYY-MM-01 format), 
        an 'ngram' column (containing single or multi-word n-grams as strings),
        and a 'count' column (frequency of that ngram for the month).
        Returns matching rows with 'month', 'ngram', and 'count'.
        """
        try:
            # Convert start_date and end_date strings to datetime objects for comparison
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            # Open the Parquet file for streaming
            table = pq.ParquetFile(file_path)

            # Initialize an empty DataFrame to collect filtered results
            filtered_data = []

            # Stream through the Parquet file in row groups
            for batch in table.iter_batches(columns=['month', 'ngram', 'count']):
                df = batch.to_pandas()

                # Ensure 'month' column is in datetime format
                df['month'] = pd.to_datetime(df['month'])

                # Filter rows based on the date range
                mask = (df['month'] >= start_dt) & (df['month'] <= end_dt)
                df_filtered = df[mask]

                # Search for the keyword within the 'ngram' column (whole word, case-insensitive)
                keyword_mask = df_filtered['ngram'].astype(str).str.contains(f'\\b{keyword}\\b', case=False, na=False, regex=True)
                filtered_data.append(df_filtered[keyword_mask])

            # Concatenate all filtered chunks
            filtered_data = pd.concat(filtered_data, ignore_index=True)

            if filtered_data.empty:
                print(f"Keyword '{keyword}' not found as a whole word in any n-gram within the specified date range.")
                return pd.DataFrame(columns=['month', 'ngram', 'count'])

            if not filtered_data.empty:
                print(self._get_ram_status())

            return filtered_data[['month', 'ngram', 'count']]
        except KeyError as e:
            print(f"Error searching for keyword: Missing expected column {e}. Ensure 'month', 'ngram', and 'count' columns exist.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error searching for keyword: {e}")
            return pd.DataFrame()

    #domains
        
    def top_domains_in_date_range(self, file_path: str, n: int, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get top domains in a specified time frame by summing their counts.
        Reads data from a Parquet file in a memory-efficient way.
        Assumes data has a 'day' column for date, 'domain' column for domain strings,
        and a 'count' column for frequency.
        """
        try:
            # Convert start_date and end_date strings to datetime objects for comparison
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            # Open the Parquet file for streaming
            table = pq.ParquetFile(file_path)

            # Initialize an empty DataFrame to collect filtered results
            filtered_data = []

            # Stream through the Parquet file in row groups
            for batch in table.iter_batches(columns=['day', 'domain', 'count']):
                df = batch.to_pandas()

                # Ensure 'day' column is in datetime format
                df['day'] = pd.to_datetime(df['day'])

                # Filter rows based on the date range
                mask = (df['day'] >= start_dt) & (df['day'] <= end_dt)
                filtered_data.append(df[mask])

            # Concatenate all filtered chunks
            filtered_data = pd.concat(filtered_data, ignore_index=True)

            if filtered_data.empty:
                print(f"No data found in the date range {start_date} to {end_date}.")
                return pd.DataFrame(columns=['domain', 'total_count'])

            # Extract domain: handles http(s):// and www. variations
            extracted_domains = filtered_data['domain'].astype(str).str.extract(r'(?:https?://)?(?:www\.)?([^/]+)')[0].str.lower()

            # Assign to a temporary column to group
            temp_df = filtered_data.assign(extracted_domain=extracted_domains)

            domain_counts = temp_df.groupby('extracted_domain')['count'].sum()

            top_n_domains = domain_counts.nlargest(n).reset_index(name='total_count').rename(columns={'extracted_domain': 'domain'})

            return top_n_domains
        except KeyError as e:
            print(f"Error processing domains: Missing expected column {e}. Ensure 'day', 'domain', and 'count' columns exist.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error processing domains: {e}")
            return pd.DataFrame()

    def keyword_search_in_domains(self, file_path: str, keyword: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Search for a keyword (case-insensitive) within extracted domains in a specified time frame.
        Reads data from a Parquet file in a memory-efficient way.
        Assumes data has a 'day' column for date, 'domain' column for domain strings,
        and a 'count' column for frequency.
        Returns matching original rows with 'day', 'domain', and 'count'.
        """
        try:
            # Convert start_date and end_date strings to datetime objects for comparison
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            # Open the Parquet file for streaming
            table = pq.ParquetFile(file_path)

            # Initialize an empty DataFrame to collect filtered results
            filtered_data = []

            # Stream through the Parquet file in row groups
            for batch in table.iter_batches(columns=['day', 'domain', 'count']):
                df = batch.to_pandas()

                # Ensure 'day' column is in datetime format
                df['day'] = pd.to_datetime(df['day'])

                # Filter rows based on the date range
                mask = (df['day'] >= start_dt) & (df['day'] <= end_dt)
                df_filtered = df[mask]

                # Extract domain for searching
                extracted_domains = df_filtered['domain'].astype(str).str.extract(r'(?:https?://)?(?:www\.)?([^/]+)')[0].str.lower()

                # Perform case-insensitive containment search on the extracted domain
                keyword_mask = extracted_domains.str.contains(keyword.lower(), case=False, na=False)

                filtered_data.append(df_filtered[keyword_mask])

            # Concatenate all filtered chunks
            filtered_data = pd.concat(filtered_data, ignore_index=True)

            if filtered_data.empty:
                print(f"Keyword '{keyword}' not found in any domain within the specified date range.")
                return pd.DataFrame(columns=['day', 'domain', 'count'])

            if not filtered_data.empty:
                print(self._get_ram_status())

            return filtered_data[['day', 'domain', 'count']]
        except KeyError as e:
            print(f"Error searching for keyword in domains: Missing expected column {e}. Ensure 'day', 'domain', and 'count' columns exist.")
            return pd.DataFrame(columns=['day', 'domain', 'count'])
        except Exception as e:
            print(f"Error searching for keyword in domains: {e}")
            return pd.DataFrame()
        
#emotions

    def plot_emotion_trends(self, file_path: str, start_date: str, end_date: str) -> None:
        """
        Plot emotion trends over a specified time range.
        Reads data from a Parquet file in a memory-efficient way.
        Assumes data has a 'date' column and columns for each emotion:
        "anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust".
        The date range must be longer than three days.
        """
        try:
            # Convert start_date and end_date strings to datetime objects for comparison
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            if start_dt >= end_dt:
                print("Error: Start date must be before end date.")
                return

            if (end_dt - start_dt).days <= 3:
                print("Error: Date range must be longer than three days.")
                return

            # Open the Parquet file for streaming
            table = pq.ParquetFile(file_path)

            # Initialize an empty DataFrame to collect filtered results
            filtered_data = []

            # Stream through the Parquet file in row groups
            for batch in table.iter_batches():
                df = batch.to_pandas()

                # Ensure 'date' column is in datetime format
                df['date'] = pd.to_datetime(df['date'])

                # Filter rows based on the date range
                mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
                filtered_data.append(df[mask])

            # Concatenate all filtered chunks
            filtered_data = pd.concat(filtered_data, ignore_index=True)

            if filtered_data.empty:
                print(f"No data found in the date range {start_date} to {end_date}.")
                return

            # Set 'date' as index for plotting
            filtered_data = filtered_data.set_index('date')

            # Plotting
            emotion_columns = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
            plt.figure(figsize=(15, 8))
            for emotion in emotion_columns:
                plt.plot(filtered_data.index, filtered_data[emotion], label=emotion.capitalize())

            plt.title(f'Emotion Trends from {start_date} to {end_date}')
            plt.xlabel('Date')
            plt.ylabel('Emotion Score / Intensity')
            plt.legend()
            plt.grid(True)

            # Format x-axis to show dates nicely
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45)
            plt.tight_layout()  # Adjust layout to prevent labels from overlapping
            plt.show()

        except KeyError as e:
            print(f"Error plotting emotion trends: Missing expected column {e}.")
        except Exception as e:
            print(f"Error plotting emotion trends: {e}")

    def plot_emotion_highlights(self, file_path: str, start_date: str, end_date: str) -> None:
        """
        Plot emotion trends, highlighting maximum and minimum non-zero values
        for each emotion over a specified time range.
        Reads data from a Parquet file in a memory-efficient way.
        Assumes data has a 'date' column and columns for each emotion.
        The date range must be longer than three days.
        """
        try:
            # Convert start_date and end_date strings to datetime objects for comparison
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            if start_dt >= end_dt:
                print("Error: Start date must be before end date.")
                return

            if (end_dt - start_dt).days <= 3:
                print("Error: Date range must be longer than three days.")
                return

            # Open the Parquet file for streaming
            table = pq.ParquetFile(file_path)

            # Initialize an empty DataFrame to collect filtered results
            filtered_data = []

            # Stream through the Parquet file in row groups
            for batch in table.iter_batches():
                df = batch.to_pandas()

                # Ensure 'date' column is in datetime format
                df['date'] = pd.to_datetime(df['date'])

                # Filter rows based on the date range
                mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
                filtered_data.append(df[mask])

            # Concatenate all filtered chunks
            filtered_data = pd.concat(filtered_data, ignore_index=True)

            if filtered_data.empty:
                print(f"No data found in the date range {start_date} to {end_date}.")
                return

            # Set 'date' as index for plotting
            filtered_data.set_index('date', inplace=True)

            plt.figure(figsize=(18, 10))  # Increased figure size for clarity

            emotion_columns = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
            line_handles = []  # For custom legend

            for emotion in emotion_columns:
                # Plot the emotion line
                line, = plt.plot(filtered_data.index, filtered_data[emotion], label=emotion.capitalize(), alpha=0.7)
                line_handles.append(line)
                line_color = line.get_color()

                # Highlight maximum value
                if not filtered_data[emotion].empty:
                    max_val = filtered_data[emotion].max()
                    if pd.notna(max_val):
                        max_date = filtered_data[emotion].idxmax()
                        plt.scatter(max_date, max_val, marker='o', color=line_color, s=80, zorder=5, edgecolors='black')

                # Highlight minimum non-zero value
                non_zero_emotion_data = filtered_data[emotion][filtered_data[emotion] > 0]
                if not non_zero_emotion_data.empty:
                    min_nz_val = non_zero_emotion_data.min()
                    min_nz_date = non_zero_emotion_data.idxmin()
                    plt.scatter(min_nz_date, min_nz_val, marker='x', color=line_color, s=80, zorder=5)

            # Create custom legend handles for markers
            from matplotlib.lines import Line2D
            legend_elements = line_handles + [
                Line2D([0], [0], marker='o', color='w', label='Max Value', markerfacecolor='gray', markeredgecolor='black', markersize=10),
                Line2D([0], [0], marker='x', color='w', label='Min Non-Zero Value', markerfacecolor='gray', markeredgecolor='black', markersize=10)
            ]

            plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))  # Place legend outside

            plt.title(f'Emotion Highlights from {start_date} to {end_date}')
            plt.xlabel('Date')
            plt.ylabel('Emotion Score / Intensity')
            plt.grid(True, linestyle='--', alpha=0.6)

            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=12))  # Adjust tick density
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for legend outside
            plt.show()

        except KeyError as e:
            print(f"Error plotting emotion highlights: Missing expected column {e}.")
        except Exception as e:
            print(f"Error plotting emotion highlights: {e}")
            
    # topics

    def top_topics_per_day_in_date_range(self, file_path: str, n: int, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get the top n topics for each date within a specified date range from a Parquet file.
        Assumes data has 'date', 'topic_name', and 'doc_count' columns.
        Returns a DataFrame with columns: 'date', 'topic_name', 'doc_count'.
        """
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            table = pq.ParquetFile(file_path)
            filtered_data = []

            for batch in table.iter_batches():
                df = batch.to_pandas()
                df['date'] = pd.to_datetime(df['date'])

                mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
                filtered_data.append(df[mask])

            filtered_data = pd.concat(filtered_data, ignore_index=True)

            if filtered_data.empty:
                print(f"No data found in the date range {start_date} to {end_date}.")
                return pd.DataFrame(columns=['date', 'topic_name', 'doc_count'])

            if 'topic_name' not in filtered_data.columns or 'doc_count' not in filtered_data.columns:
                print("Error: Required columns not found.")
                return pd.DataFrame(columns=['date', 'topic_name', 'doc_count'])

            top_topics = (
                filtered_data
                .sort_values(['date', 'doc_count'], ascending=[True, False])
                .groupby('date')
                .head(n)
                .reset_index(drop=True)
            )

            if not top_topics.empty:
                print(self._get_ram_status())

            return top_topics[['date', 'topic_name', 'doc_count']]
        except KeyError as e:
            print(f"Error processing topics: Missing expected column {e}.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error processing topics: {e}")
            return pd.DataFrame()

    def search_keyword_in_topics_by_date(self, file_path: str, keyword: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Search for a keyword (case-insensitive, whole word) in the 'top_words' column within a Parquet file.
        Return topic clusters (by date) where that keyword occurred.
        Assumes data has 'date', 'topic_name', 'top_words', and 'doc_count' columns.
        Returns a DataFrame with columns: 'date', 'topic_name', 'top_words', 'doc_count'.
        """
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            table = pq.ParquetFile(file_path)
            filtered_data = []

            for batch in table.iter_batches():
                df = batch.to_pandas()
                df['date'] = pd.to_datetime(df['date'])

                mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
                filtered_data.append(df[mask])

            filtered_data = pd.concat(filtered_data, ignore_index=True)

            if filtered_data.empty:
                print(f"No data found in the date range {start_date} to {end_date}.")
                return pd.DataFrame(columns=['date', 'topic_name', 'top_words', 'doc_count'])

            for col in ['topic_name', 'top_words', 'doc_count']:
                if col not in filtered_data.columns:
                    print(f"Error: '{col}' column not found.")
                    return pd.DataFrame(columns=['date', 'topic_name', 'top_words', 'doc_count'])

            keyword_mask = filtered_data['top_words'].astype(str).str.contains(f'\\b{keyword}\\b', case=False, na=False, regex=True)
            keyword_data = filtered_data[keyword_mask]

            if keyword_data.empty:
                print(f"Keyword '{keyword}' not found in any topic cluster within the specified date range.")
                return pd.DataFrame(columns=['date', 'topic_name', 'top_words', 'doc_count'])

            if not keyword_data.empty:
                print(self._get_ram_status())

            return keyword_data[['date', 'topic_name', 'top_words', 'doc_count']]
        except KeyError as e:
            print(f"Error searching for keyword in topics: Missing expected column {e}.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error searching for keyword in topics: {e}")
            return pd.DataFrame()

    def top_subreddits_for_keyword_by_date(self, file_path: str, keyword: str, n: int, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Return the top n subreddit_ids for a keyword search in 'top_words', sorted by date and doc_count, from a Parquet file.
        Assumes data has 'date', 'subreddit_id', 'top_words', and 'doc_count' columns.
        Returns a DataFrame with columns: 'date', 'subreddit_id', 'top_words', 'doc_count'.
        """
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            table = pq.ParquetFile(file_path)
            filtered_data = []

            for batch in table.iter_batches():
                df = batch.to_pandas()
                df['date'] = pd.to_datetime(df['date'])

                mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
                filtered_data.append(df[mask])

            filtered_data = pd.concat(filtered_data, ignore_index=True)

            if filtered_data.empty:
                print(f"No data found in the date range {start_date} to {end_date}.")
                return pd.DataFrame(columns=['date', 'subreddit_id', 'top_words', 'doc_count'])

            for col in ['subreddit_id', 'top_words', 'doc_count']:
                if col not in filtered_data.columns:
                    print(f"Error: '{col}' column not found.")
                    return pd.DataFrame(columns=['date', 'subreddit_id', 'top_words', 'doc_count'])

            keyword_mask = filtered_data['top_words'].astype(str).str.contains(f'\\b{keyword}\\b', case=False, na=False, regex=True)
            keyword_data = filtered_data[keyword_mask]

            if keyword_data.empty:
                print(f"Keyword '{keyword}' not found in any subreddit within the specified date range.")
                return pd.DataFrame(columns=['date', 'subreddit_id', 'top_words', 'doc_count'])

            top_subreddits = (
                keyword_data
                .sort_values(['date', 'doc_count'], ascending=[True, False])
                .groupby('date')
                .head(n)
                .reset_index(drop=True)
            )

            if not top_subreddits.empty:
                print(self._get_ram_status())

            return top_subreddits[['date', 'subreddit_id', 'top_words', 'doc_count']]
        except KeyError as e:
            print(f"Error searching for keyword in subreddits: Missing expected column {e}.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error searching for keyword in subreddits: {e}")
            return pd.DataFrame()
        
    def _get_ram_status(self):
        process_mem = psutil.Process(os.getpid())
        mem_mb = process_mem.memory_info().rss / (1024 * 1024)
        total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)  # Dynamically fetch total system memory in MB
        percent = (mem_mb / total_memory_mb) * 100

        status = f"📊 Memory Usage: {mem_mb:.1f}MB / {total_memory_mb:.1f}MB ({percent:.1f}%)"
        if percent > 85:
            status += " ⚠️ WARNING: High usage! Please narrow your search."
        return status        


class NotebookGuard:
    def __init__(self):
        self.unlocked = False

    def display_gate(self):
        button = widgets.Button(
            description="🔓 Initialize Environment",
            button_style='success', # Green button
            tooltip='Click to enable data loading'
        )
        output = widgets.Output()

        def on_click(b):
            self.unlocked = True
            with output:
                output.clear_output()
                print("✅ Ready! You can now run the cells below manually.")
            button.disabled = True

        button.on_click(on_click)
        display(button, output)

    def check(self):
        if not self.unlocked:
            # This is the "Stop Sign" for Run All
            assert self.unlocked, "🛑 STOP: Click the 'Initialize' button above before running this cell."