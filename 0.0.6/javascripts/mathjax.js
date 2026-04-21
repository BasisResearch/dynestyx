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
        MathJax.startup.output.clearCache()
        MathJax.typesetClear()
        MathJax.texReset()
        MathJax.typesetPromise()
    })
}