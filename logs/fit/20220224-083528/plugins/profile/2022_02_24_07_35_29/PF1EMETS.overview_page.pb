?&$	9??%????$?? ?V???St$???!4??ؙ??$	?????G@%??O?^???֔Vw?@!?,???@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails04??ؙ??:w?^????A?0?????Y2;?ީ??rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?ԱJ????_????A?{G?	1??Yfg?;p??rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0? ??z????+,????A??[?t??Y???/?^??rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0G?P?[??0*??D??A?,`????Y????)??rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?۞ ?]??DkE?????A??@I???Y?y?Տ??rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?St$???,?j????At
??????Y????(??rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0r6ܬ???a??????A?ۼqR??Yτ&?%??rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	????v??in??K??A?g#?M)??Yt?%z???rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
?Y??B??????G6W??A}?;l"3??Y?xx??rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???T???????il??AZ??c!??Y1[?*?M??rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?z??>??_9?????Aa⏢????Y??@fgћ?rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0,?-X*???k|&????AM??E??Y?[????rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??Z??????g?R??A??2W??Y????ׁ??rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??6????3??7???A&p?n???Y6w??\???rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0R?GT????%?L1??A??߼8???Y_?lW胕?rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails08?πz??? {??????AF????(??Y??~???rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?w????(?1k??A?N"¿??Ykg{????rtrain 97*	)\????@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?dT8??!V?????C@)$???9"??1?E?C2?A@:Preprocessing2E
Iterator::Root>$|?o???!@.????@@)S?'??Z??1`??U41@:Preprocessing2T
Iterator::Root::ParallelMapV2)_?BF??! ?XE?00@))_?BF??1 ?XE?00@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceiT?d??! ????"@)iT?d??1 ????"@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??????!?h7???P@)?1???A??1?t???"@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?c?? w??!?n?O?+@){???w???1?)?G!?@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap,??????!e:b???1@)mU???1?2W!?@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*Z???а??!???I@)Z???а??1???I@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 46.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9H?ۮu=@I?!?R&X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??%???????vȱ?,?j????!:w?^????	!       "	!       *	!       2$	??????AάZE???N"¿??!?0?????:	!       B	!       J$	?????????????p?????ׁ??!2;?ީ??R	!       Z$	?????????????p?????ׁ??!2;?ީ??b	!       JCPU_ONLYYH?ۮu=@b q?!?R&X@Y      Y@qɜ?`}?R@"?	
both?Your program is POTENTIALLY input-bound because 46.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?75.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 