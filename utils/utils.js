const removeWordsFromStartAndAfter = (inputString, targetWord) => {
    const targetIndex = inputString.indexOf(targetWord);
        if (targetIndex !== -1) {
            const wordsToKeep = inputString.substring(targetIndex+ 13).trim();
            return wordsToKeep;
        }
        return inputString;
}

module.exports = {removeWordsFromStartAndAfter}