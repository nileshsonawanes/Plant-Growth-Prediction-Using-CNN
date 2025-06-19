var sectionArray = [1, 2, 3];

var isClickScrolling = false;

var $allNavLinks = $('.navbar-nav .nav-link');
var $clickScrollLinks = $('.navbar-nav .nav-link.click-scroll');

$(document).scroll(function(){
    if(isClickScrolling) return; // Ignore scroll events during click scroll animation
    
    var docScroll = $(document).scrollTop();
    var offsetSection;
    var activeIndex = -1;

    var firstSectionOffset = $('#' + 'section_' + sectionArray[0]).offset().top - 100;

    if(docScroll < firstSectionOffset){
        // If scrolled above first section, activate first nav link
        $allNavLinks.removeClass('active').addClass('inactive');
        $clickScrollLinks.eq(0).addClass('active').removeClass('inactive');
        return;
    }

    // Iterate sections in reverse order to find the current section
    for(var i = sectionArray.length - 1; i >= 0; i--){
        offsetSection = $('#' + 'section_' + sectionArray[i]).offset().top - 100; // Adjusted offset for navbar height
        if(docScroll >= offsetSection){
            activeIndex = i;
            break;
        }
    }

    if(activeIndex === -1){
        $allNavLinks.removeClass('active').addClass('inactive');
        $clickScrollLinks.eq(0).addClass('active').removeClass('inactive');
    } else {
        $allNavLinks.removeClass('active').addClass('inactive');
        $clickScrollLinks.eq(activeIndex).addClass('active').removeClass('inactive');
    }
});

$.each(sectionArray, function(index, value){
    $clickScrollLinks.eq(index).click(function(e){
        var offsetClick = $('#' + 'section_' + value).offset().top - 100; // Adjusted offset for navbar height
        e.preventDefault();
        isClickScrolling = true;
        $('html, body').animate({
            'scrollTop':offsetClick
        }, 300, function(){
            isClickScrolling = false; // Re-enable scroll events after animation
        });
        // Fix highlight on click
        $allNavLinks.removeClass('active').addClass('inactive');
        $clickScrollLinks.eq(index).addClass('active').removeClass('inactive');
    });
});

$(document).ready(function(){
    $('.navbar-nav .nav-item .nav-link:link').addClass('inactive');    
    $clickScrollLinks.eq(0).addClass('active').removeClass('inactive');
});
