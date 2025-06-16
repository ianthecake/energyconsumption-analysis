# Befehlsübersicht
Eine kurze Übersicht über die wichtigsten Befehle die in der Vorlage enthalten sind.
Erstmalige Einstellung (Titel, Name, Studienjahr, etc.) in der vorlage.tex
Dort ist angegeben, wo ihr die Änderung vornehmen müsst.

### Überschriften und Gliederung

```TeX
\section{erste Gliederungsebene}
\subsection{zweite Gliederungsebene}
\subsubsection{dritte Gliederungsebene}

\newline    % neue Zeile
\newpage    % neue Seite
\clearpage  % neue Seite und vorherige Formatierung löschen
\noindent   % kein Einzug bei neuer Zeile

\footnote{Hier kannst du deine Fußzeile reinschreiben}
```

### Text Style
```TeX
\gqq{Anführungszeichen}

\textbf{Fettgedruckt}
\textit{Kursiv}		
\underline{Unterstrichen}

% Manueller Zeilenumbruch
\\

% Für längere Zitate geeignet - versch. Beispiele nachfolgend:
\myboxquote{Dieser Text steht in einer grauen Box}
\myboxquote{\textit{\gqq{ TEXT HIER! \cite[vgl.][S.]{Quelle} } } }

% links Anführungszeichen
\glqq

% rechts Anführungszeichen
\grqq{}
```

### Refernzen und Verweise

```TeX
\label{referenzKey}     % Einen Referenzpunkt setzen
\ref{referenzKey}       % auf den referenzpunkt verweisen
\mypageref{referenzKey} % auf den referenzpunkt verweisen mit Seitenangabe
```
### Aufzählungen
Stichpunkte:
```TeX
\begin{itemize}						% Beginn einer Aufzählung
	\item erster Punkt				% Aufzählungspunkt
	\item Zweiter Punkt				% Aufzählungspunkt
\end{itemize}						% Ende der Aufzählung
```

Aufzählung:
```TeX
\begin{enumerate}					% Beginn einer Aufzählung
	\item erster Punkt				% Aufzählungspunkt
	\item Zweiter Punkt				% Aufzählungspunkt
\end{enumerate}						% Ende der Aufzählung
```

### Bilder
für genauere Erläuterung siehe hier:

```TeX
% Folgende Struktur für individuelle Anpassung der Seitenverhältnisse:

\begin{figure}[hbt]							% Beginn einer Grafik
	\centering 								% trim=left bottom right top
	\includegraphics[trim = 200mm 0mm 0mm 0mm, width=0.5\textwidth]{images/jubilaeum.jpg}
	\caption{Bild beschriftung} {\emph{\vgl[540]{BiBkey}}}
	\label{jubilaeum}
\end{figure}

\begin{figure}[hbt]			
    \centering
    \includegraphics[width=15cm]{images/x.jpg}
    \caption{Bildunterschrift} \cite[vgl.][S. 25]{Bibkey}
    \label{fig:abb3.3}
\end{figure}

\begin{figure}[h!]
    \centering
    \includegraphics[width=14cm]{images/x.png}
    \caption{Bildunterschrift}\cite[vgl.][S. 35]{Bibkey}
    \label{fig:abb2.7}
\end{figure}

Zum Floating (Position) der Bilder ist der Wert in 
\begin{figure}[HIER DIESER WERT] eckigen Klammern wichtig.
Folgende Werte können verwendet werden:

h	Place the float here (nicht genau aber so ungefähr!)
t	Position at the top of the page.
b	Position at the bottom of the page.
p	Put on a special page for floats only.
!	Override internal parameters LaTeX uses for determining "good" float positions
H	Places the float at precisely the location in the LATEX code. Requires the float            package. This is somewhat equivalent to h!.
h!  Place the float exactly on the position where placed!
hbt Overleaf versucht selbst das Bild bestmöglich auszurichten

```

### Tabelle
```TeX
\begin{table}[hbt]
    \centering
    \caption{Beschriftung {\emph{\vgl[540]{BiBkey}}}
    \label{big5-korrelation}
	--> Tabelle einfügen. Als LaTeX tabelle oder Bild
\end{table}
```

### Glossar
```TeX
\gls{BPM} 		% Glossar eintrag
\glspl{BPM} 	% Plural des Glossar eintrags
\gls{ASM} 		% Wenn Glossar eintrag erstamlig verwendet wird
\glspl{ASM} 	% Bezeichnung ausgeschrieben, anschließend wird abgekürzt
```

### Literaturquellen

```TeX
\vgl{BiBkey} 					% Vergleichs verweis ohne Seitenangabe
\vgl[2]{BiBkey} 				% Vergleichs verweis mit Seitenangabe
\vgl[13,~46]{BiBkey}			% Vergleichs verweis mit mehreren Seitenangaben

\vglb{BiBkey} 					% Vergleichs Verweis in Klammer ohne Seitenangabe
\vglb[2]{BiBkey} 				% Vergleichs Verweis in Klammer mit Seitenangabe
\vglb[13,~46]{BiBkey}			% Vergleichs Verweis in Klammer mit mehreren Seitenangaben

\cite{BiBkey} 					% Direkter Literatur Aufruf
\cite[S. 25]{Bibkey}            % Verweis mit Seitenangabe in Klammer
\cite[vgl.][S. 25]{Bibkey}      % Vergleichsverweis mit Seitenangabe in Klammer

\citet{Bibkey}                  % Direkter Verweis ohne Seite
\citet[S. 25]{Bibkey}           % Direkter Verweis mit Seite
```