# This is a Google Sketchup plugin for generated images for various angles

require 'sketchup.rb'

Sketchup.send_action "showRubyPanel:"

UI.menu("Plugins").add_item("Generate data") {


  # Set up rendering style 
  styles = Sketchup.active_model.styles
  
  puts styles["gustav"].class
  style = styles["gustav"]
  if style.instance_of?(Sketchup::Style) then
    styles.selected_style = style
  else
    styles.add_style "/Users/slimgee/git/vision-research/cad/sketchup/gustav.style", true
  end

  si = 128 
  model = Sketchup.active_model
  view = model.active_view
  N = 40 

  range = (0..N).map { |i| -1 + 2.0 * i/N.to_f } 
  # step(0.1) does not work in SketchUp's Ruby version
  i = 0


  range.each do |x1|
    range.each do |x2|
      both = x1**2 + x2**2
      if 1.0 - both >= 0.0 then 
        #puts "processing #{x1} and #{x2}"
        #angle = 2.0 * Math::PI * x / N 
        #eye = [700 * Math.sin(angle), 700 * Math.cos(angle), 150]
        c = Math.sqrt(1.0 - both)
        #puts "="*30,x1, x2
        x = 140 * 2 * x1 * c
        y = 140 * 2 * x2 * c 
        z = 140 * (1 - 2 * both)
        if z >= 0 then
          eye = [x, y, z]
          target = [0, 0, 0]
          up= [0, 0, 1]
          filename = "/Users/slimgee/git/data/bike/bicycle1_#{i}.png"
          if x != 0 or y != 0 then
            if not File.exists? filename then
              camera = Sketchup::Camera.new eye, target, up
              view.camera=camera
              keys = {
                :filename => filename,
                :width => si,
                :height => si,
                :antialias => true,
                :compression => 0.9,
                :transparent => true,
              }
              status = view.write_image keys 
            end
            i += 1
          end
        end
      end
    end
  end 
  UI.messagebox("Data saved! (#{i})")
}

