# This is a Google Sketchup plugin for generated images for various angles

require 'sketchup.rb'

Sketchup.send_action "showRubyPanel:"

def gaussian(mean, stddev)
    if stddev == 0.0 then
        return mean
    end
    theta = 2 * Math::PI * rand()
    rho = Math.sqrt(-2 * Math.log(1 - rand()))
    scale = stddev * rho
    x = mean + scale * Math.cos(theta)
    y = mean + scale * Math.sin(theta)
    return [mean+stddev*2, [mean-stddev*2, x].max].min
end



UI.menu("Plugins").add_item("Generate uniform") {

    #prompts = ['Name of this model?']
    #defaults = ['carN']
    #input = UI.inputbox prompts, defaults, 'Give us some info'

    # Set up rendering style 
    styles = Sketchup.active_model.styles
    srand(0)
    
    puts styles["gustav"].class
    style = styles["gustav"]
    if style.instance_of?(Sketchup::Style) then
        styles.selected_style = style
    else
        styles.add_style "/Users/slimgee/git/vision-research/cad/sketchup/gustav.style", true
    end

    model = Sketchup.active_model
    model.entities.each do |entity|
        puts entity 
        if entity.is_a? Sketchup::Group
            puts 'This is a group'
            puts entity.name
            entity.visible = false 
        end
    end

    SD = 0.0

    model.entities.each do |entity|
        if entity.is_a? Sketchup::Group
            name = entity.name
            entity.visible = true
            
            view = model.active_view
            puts('view', view)
            si = 256 

            save_image = Proc.new { |viewname, i, altitude, azimuth, out_of_plane, target, dist0, focal_length|
                dist = dist0 * focal_length / 67.0
                #rotation = outofplane_index * 6.0 * Math::PI / 180.0
                x = dist * Math.sin(Math::PI / 2.0 - altitude) * Math.cos(azimuth)
                y = dist * Math.sin(Math::PI / 2.0 - altitude) * Math.sin(azimuth) 
                z = dist * Math.cos(Math::PI / 2.0 - altitude) 

                eye = [x + target[0], y + target[1], z + target[2]]
                up= [
                    Math.sin(out_of_plane) * Math.cos(azimuth + Math::PI / 2.0), 
                    Math.sin(out_of_plane) * Math.sin(azimuth + Math::PI / 2.0), 
                    Math.cos(out_of_plane)
                ]
                #filename = "/Users/slimgee/git/data/xi3zao3-car-sides7/view%03d_%s.png" % [i, name]
                filename = "/Users/slimgee/git/data/sketchup-output/%s%03d_%s.png" % [viewname, i, name]
                if x != 0 or y != 0 then
                    if not File.exists? filename then
                        camera = Sketchup::Camera.new eye, target, up
                        #puts 'perspective', camera.perspective?
                        camera.perspective = true
                        camera.focal_length = focal_length
                        #puts 'Camera focal length'
                        #puts camera.focal_length
                        view.camera = camera
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
                    true
                else
                    false
                end 
            } 

            #dist = 140 # Bike

            N = 36 
            angles = (0...N).map { |i| 360 * i/N.to_f } 
            M = 3
            altitudes = (0...M).map { |i| 20 * i }
            # step(0.1) does not work in SketchUp's Ruby version
            dist = 500 
            dist2 = 330  # car
            dist3 = 450

            i = 0
            N = 30

            # half-side cars
            if true 
                flip1 = flip2 = 0
                #[[25, 3.5], [50, 25]].each do |angle, altitude0|
                angles.each do |angle|
                    altitudes.each do |altitude0|
                        [40, 65, 80].each do |focal_length|
                            focal_length += gaussian(0, 8 * SD)
                            rand_altitude = gaussian(0, 3 * SD)
                            altitude = (altitude0 + rand_altitude) * Math::PI / 180.0
                            #azimuth = (-45 + 90 * flip + (rand() - 0.5) * 7.5) * Math::PI / 180.0      #sides1
                            #azimuth = (-45 + 90 * flip + (rand() - 0.5) * 20.0) * Math::PI / 180.0     #sides2
                            rand_azimuth = gaussian(0, 10 * SD)
                            azimuth = (90 + angle * (2*flip1-1) + 180 * flip2 + rand_azimuth) * Math::PI / 180.0
                            rand_out_of_plane = gaussian(0, 2.5 * SD)
                            out_of_plane = rand_out_of_plane * Math::PI / 180.0
                            if save_image.call("view", i, altitude, azimuth, out_of_plane, [0.0, 0, 0], dist3, focal_length)
                                i += 1
                            end
                        end
                    end
                end
            end

            entity.visible = false 
        end
    end
    UI.messagebox("Data saved!")
}

