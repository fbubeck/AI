?($	??????b?H?????:pΈ???!????o??$	???.4@?ů#??
@[??ۮm@!???M?1@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?&1???3ı.n???At$???~??Y??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&4??7???????o_??A䃞ͪ???Y	??g????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????o??C?i?q???AaTR'????Y?w??#???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?h o????+e?X??A`??"????Y?f??j+??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&$???~????!?uq??A?H?}8??Y-C??6??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&O@a????Ԛ?????A??镲??Y?A`??"??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???K7????%䃞???AC?i?q???Y???S㥛?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ͪ??V?????ׁs??A?T???N??Y???Q???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Q???x$(~???AO@a????YA??ǘ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	i o????!?rh????Aŏ1w-!??Y??(????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
mV}??b??6?>W[???A?v??/??Y???_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&gDio????&䃞ͪ??A???_vO??Y?!??u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&!?rh????2U0*???A??ZӼ???Y??Pk?w??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&1?Zd??M?J???Aj?t???Y?0?*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&yX?5?;????9#J{??A=?U????Y??ݓ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ș?????a2U0*???A?ڊ?e???Y?#??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&m???{??????Q???A3ı.n???Yy?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?O??n??ꕲq???A???????Y???Q???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?b?=y??K?46??A???????YK?=?U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??3??????6???A?a??4???Yy?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?:pΈ?????m4????A?}8gD??Y???QI??*	??????@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?ʡE????!?]?dցA@)HP?s??1d????>@:Preprocessing2F
Iterator::Model~8gDi??!????"?B@)tF??_??1??Jo?7@:Preprocessing2U
Iterator::Model::ParallelMapV2鷯???![\b??/,@)鷯???1[\b??/,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip$???????!nB
?O@){?/L?
??1i?g?~%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceTt$?????!u?Cs?@)Tt$?????1u?Cs?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?%䃞ͺ?!D!m[?#*@)
ףp=
??1J?Cx@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?5?;Nѡ?!L֪?H`@)?5?;Nѡ?1L֪?H`@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap	?^)???!?ة??`0@)?A`??"??1}@?O?v
@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 48.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9????L[@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	???#?-?????e;????m4????!!?rh????	!       "	!       *	!       2$	,?[?L????^?F?_???}8gD??!aTR'????:	!       B	!       J$	F&^a?????Ҍ???Pk?w??!??????R	!       Z$	F&^a?????Ҍ???Pk?w??!??????JCPU_ONLYY????L[@b Y      Y@qu???/}U@"?
both?Your program is POTENTIALLY input-bound because 48.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?85.956% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 