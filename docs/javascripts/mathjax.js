window.MathJax = {
    tex: {
        inlineMath: [["\\(", "\\)"], ["$", "$"]],
        displayMath: [["$$", "$$"], ["\\[", "\\]"]],
        processEscapes: true,
        processEnvironments: true,
        packages: { '[+]': ['ams'] }
    },
    options: {
        // Commented out to make Jupyter math work
        // ignoreHtmlClass: ".*|", 
        processHtmlClass: "arithmatex"
    }
};

if (typeof document$ !== 'undefined') {
    document$.subscribe(() => {
        if (typeof MathJax.typesetClear === 'function') {
            MathJax.typesetClear()
        }
        if (typeof MathJax.texReset === 'function') {
            MathJax.texReset()
        }
        if (typeof MathJax.typesetPromise === 'function') {
            MathJax.typesetPromise()
        }
    })
}
